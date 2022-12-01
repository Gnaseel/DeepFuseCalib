import random
import numpy as np
import argparse
import torch
import os
from torch import Tensor
import torch.utils.data
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.nn import Module, MSELoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
from datetime import datetime
import shutil
from net.CalibrationNet import get_model
from dataset import TrainDataset as CalibDataset, run_stat
from utils import count_parameters, inv_transform_vectorized, phi_to_transformation_matrix_vectorized
from utils import lidar_projection, inv_transform, phi_to_transformation_matrix, merge_color_img_with_depth, calib_np_to_phi

###################
import sys
sys.path.append('/workspace/src/src/depth')
# print(f"PATH {sys.path}")
# sys.path.append('/workspace/src/src')
# sys.path.append('/workspace/src')

from dataloaders.kitti_loader import load_calib, input_options, KittiDepth
# from metrics_d import AverageMeter, Result
import criteria_d
import helper
import vis_utils

from model import ENet
from model import PENet_C1_train
from model import PENet_C2_train
#from model import PENet_C4_train (Not Implemented)
from model import PENet_C1
from model import PENet_C2
from model import PENet_C4

import CoordConv

from dataloaders import transforms
from torchvision.utils import save_image
from PIL import Image
import sys
# set seeds
SEED = 53
rng = np.random.default_rng(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# DEVICE = 'cuda'
DEVICE = 'cuda:0'


class Loss(Module):
    def __init__(self, dataset: CalibDataset, reduction: str = 'mean',
                 alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0):
        super(Loss, self).__init__()
        self.dataset = dataset

        self.mse_loss_fn_r = MSELoss(reduction=reduction)
        self.mse_loss_fn_t = MSELoss(reduction=reduction)
        self.center_loss_fn = MSELoss(reduction=reduction)
        self.depth_loss_fn = MSELoss(reduction=reduction)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        return

    def forward(self, pred: Tuple[Tensor, Tensor], target: Tuple[Tensor, Tensor],
                velo: Tensor, T: Tensor):
        min_pts = []
        rot_pred_, trans_pred_ = self.dataset.destandardize(*pred)
        rot_err_, trans_err_ = self.dataset.destandardize(*target)

        T_err = phi_to_transformation_matrix_vectorized(rot_err_, trans_err_, device=DEVICE)
        T_fix = inv_transform_vectorized(phi_to_transformation_matrix_vectorized(rot_pred_, trans_pred_, device=DEVICE), device=DEVICE)
        T_recalib = T_fix.bmm(T_err.bmm(T))

        # reproject velo points
        pts3d_cam_l = []
        pts3d_cam_recalib_l = []
        for i in range(velo.shape[0]):
            scan = velo[i]
            T_ = T[i]
            T_recalib_ = T_recalib[i]
            # Reflectance > 0
            pts3d = scan[scan[:, 3] > 0, :]
            pts3d[:, 3] = 1
            min_pts.append(pts3d.shape[0])
            # project
            pts3d_cam = T_.mm(pts3d.t()).t().unsqueeze(0)
            pts3d_cam_recalib = T_recalib_.mm(pts3d.t()).t().unsqueeze(0)
            pts3d_cam_l.append(pts3d_cam)
            pts3d_cam_recalib_l.append(pts3d_cam_recalib)

        min_pts = min(min_pts)
        pts3d_cam_l = [pts[:, :min_pts, :] for pts in pts3d_cam_l]
        pts3d_cam_recalib_l = [pts[:, :min_pts, :] for pts in pts3d_cam_recalib_l]
        pts3d_cam = torch.cat(pts3d_cam_l, dim=0)
        pts3d_cam_recalib = torch.cat(pts3d_cam_recalib_l, dim=0)

        loss_mse = self.mse_loss(pred, target)
        loss_center = self.center_loss(pts3d_cam_recalib, pts3d_cam)
        loss_depth = self.depth_loss(pts3d_cam_recalib, pts3d_cam)
        return loss_mse + loss_center + loss_depth

    def mse_loss(self, pred: Tuple[Tensor, Tensor], target: Tuple[Tensor, Tensor]):
        rot, trans = pred
        gt_r, gt_t = target
        loss_r = self.mse_loss_fn_r(rot, gt_r)
        loss_t = self.mse_loss_fn_t(trans, gt_t)
        return self.alpha * (loss_r + loss_t)

    def center_loss(self, pts_recalib: Tensor, pts_orig: Tensor):
        c_orig = pts_orig[:, :, :3].mean(dim=1)
        c_recalib = pts_recalib[:, :, :3].mean(dim=1)
        return self.beta * self.center_loss_fn(c_recalib, c_orig)

    def depth_loss(self, pts_recalib: Tensor, pts_orig: Tensor):
        return self.gamma * self.depth_loss_fn(pts_recalib, pts_orig)

def crop_img(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """bottom-center cropping (crop the sky part and keep center)"""
    assert len(img.shape) >= 2, 'img must be a shape of (H, W) or (H, W, C)'
    height, width = img.shape[2], img.shape[3]
    top_margin = int(height - h)
    left_margin = int((width - w) / 2)

    if len(img.shape) == 3:
        image = img[top_margin:top_margin + h, left_margin:left_margin + w, :]
    else:
        image = img[:,:,top_margin:top_margin + h, left_margin:left_margin + w]

    return image

def train_one_epoch( model_d, model: Module, loss_fn: Loss, opt: Optimizer, loader: DataLoader,
                    epoch: int, global_i: int, logger: SummaryWriter):
    running_loss = 0
    running_mean = 0
    pbar = tqdm(loader)
    model.train()
    model_d.eval()

    for i, sample in enumerate(pbar):

        cam2_d, velo_d, gt_err, gt_err_norm, cam2, velo, T, P = sample[0].to(DEVICE), sample[1].to(DEVICE), sample[2], \
                                                       sample[3].to(DEVICE), sample[4].to(DEVICE), sample[5].to(DEVICE), sample[6].to(DEVICE), sample[7].to(DEVICE)
        # print(f"shape {cam2_d.shape}")
        # CROP!!
        with torch.no_grad():

            cam2_d = crop_img(cam2_d, 352, 1216)
            cam2 = cam2[0].cpu().numpy()            
            velo_np = velo[0].cpu().numpy()
            T_np = T[0].cpu().numpy()
            P = P[0].cpu().numpy()
            gt_err = gt_err.numpy()
            T_err = phi_to_transformation_matrix(gt_err[0])
            T_composed = T_err @ T_np
            phi = T_composed.copy()
            loss_phi = np.eye(4, dtype=np.float32)
            # print(f"shape {cam2_d.shape}")
            assert cam2_d.shape[-2:] == velo_d.shape[-2:], f'model input shape mismatch. ' \
                                                           f'Input 1 shape: {cam2_d.shape[-2:]}. ' \
                                                           f'Input 2 shape: {velo_d.shape[-2:]}'


            # Eval -------------------------- Depth
            # M_VELOD = 14.0833
            # S_VELOD = 8.7353
            # velo_d = (velo_d - M_VELOD) / S_VELOD

            M_VELOD = 14.0833
            S_VELOD = 8.7353
            velo_d = (velo_d * S_VELOD)+ M_VELOD
            # print(f'Cam2 shape {cam2.shape}, Velo shape {velo_d.shape}')

            cam2_tensor = np.transpose(cam2, (2, 0, 1)) 
            cam2_tensor = torch.Tensor(cam2_tensor)
            cam2_tensor = cam2_tensor.unsqueeze(0)
            cam2_tensor = crop_img(cam2_tensor, 352, 1216)
            batch_data = {"rgb": cam2_tensor, "d": velo_d,\
              'position': args.position, 'K': args.K}
            batch_data = {
                key: val.to(DEVICE)
                for key, val in batch_data.items() if val is not None
            }
            pred_d = model_d(batch_data)
            # print(f"Data {pred_d.shape}  {torch.mean(pred_d)} {torch.max(pred_d)} {torch.min(pred_d)}")
            pred_d = 80 - pred_d
            # print(f"AfTER Data {pred_d.shape}  {torch.mean(pred_d)} {torch.max(pred_d)} {torch.min(pred_d)}")
            pred_d = torch.where(pred_d>0, pred_d, torch.tensor([0.]).to(DEVICE))
            pred_d = (pred_d - M_VELOD) / S_VELOD
            # print(f"Shape of pred {pred_d.shape}")

            # forward -------------------------- CUR
            # pred_r, pred_t = model(velo_d, cam2_d)

        pred_r, pred_t = model(pred_d, cam2_d)
        loss = loss_fn((pred_r, pred_t), torch.split(gt_err_norm, 3, dim=1), velo, T)


        # backward
        # zero the parameter gradients
        opt.zero_grad()
        # auto-calculate gradients
        loss.backward()
        # apply gradients
        opt.step()
        # depth_path = os.path.join(args.data_folder_save, str(i).zfill(10) + '_output.png')
        # rgb_path = os.path.join(args.data_folder_save, str(i).zfill(10) + '_output_rgb.png')
        # print(depth_path)
        # vis_utils.save_depth_as_uint16png_upload(pred_d, depth_path)
        # im = Image.fromarray(cam2)
        # im.save(rgb_path)

        # collect statistics
        running_loss += loss.item()
        running_mean = running_loss / (i + 1)
        pbar.set_description("Epoch %d, train running loss: %.4f" % (epoch + 1, running_mean))

        # log statistics
        if (i+1) % 100 == 0:
            gt_splits = np.split(gt_err, 2, axis=1)  # split into 2 parts
            pred_r, pred_t = loader.dataset.destandardize(pred_r.detach().cpu().numpy(), pred_t.detach().cpu().numpy())
            err_a, err_b, err_c = np.abs(pred_r-gt_splits[0]).mean(axis=0)
            err_x, err_y, err_z = np.abs(pred_t-gt_splits[1]).mean(axis=0)
            logger.add_scalar('Train/Loss/loss', running_mean, global_i)
            logger.add_scalar('Train/Loss/roll', err_a, global_i)
            logger.add_scalar('Train/Loss/pitch', err_b, global_i)
            logger.add_scalar('Train/Loss/yaw', err_c, global_i)
            logger.add_scalar('Train/Loss/x', err_x, global_i)
            logger.add_scalar('Train/Loss/y', err_y, global_i)
            logger.add_scalar('Train/Loss/z', err_z, global_i)
        global_i += 1
    return running_mean, global_i


def validation(model_d, model: Module, loss_fn: Loss, loader: DataLoader,
               epoch: int, global_i: int, logger: SummaryWriter):
    running_loss = 0
    running_mean = 0
    rot_err = np.zeros(3, np.float32)
    trans_err = np.zeros(3, np.float32)
    pbar = tqdm(loader)
    model.eval()

    with torch.no_grad():
        i = 0
        for i, sample in enumerate(pbar):
            
            cam2_d, velo_d, gt_err, gt_err_norm, cam2, velo, T, P = sample[0].to(DEVICE), sample[1].to(DEVICE), sample[2], \
                                                       sample[3].to(DEVICE), sample[4].to(DEVICE), sample[5].to(DEVICE), sample[6].to(DEVICE), sample[7].to(DEVICE)
            # cam2_d, velo_d, gt_err, gt_err_norm, velo, T = sample[0].to(DEVICE), sample[1].to(DEVICE), sample[2], \
            #                                                sample[3].to(DEVICE), sample[4].to(DEVICE), sample[5].to(DEVICE)
            cam2_d = crop_img(cam2_d, 352, 1216)
            cam2 = cam2[0].cpu().numpy()            
            velo_np = velo[0].cpu().numpy()
            T_np = T[0].cpu().numpy()
            P = P[0].cpu().numpy()
            gt_err = gt_err.numpy()
            T_err = phi_to_transformation_matrix(gt_err[0])
            T_composed = T_err @ T_np
            phi = T_composed.copy()
            loss_phi = np.eye(4, dtype=np.float32)
    
            # Eval -------------------------- Depth
            # print(f'---Cam2 shape {cam2.shape}, Velo shape {velo_d.shape}')
            # print(f'---Cam2 shape {cam2.shape}, Velo shape {velo_d.shape}')
            # print(f"Data {velo_d[0].shape}  {torch.mean(velo_d[0])} {torch.max(velo_d[0])} {torch.min(velo_d[0])}")
            # print(f"Data {velo_d[1].shape}  {torch.mean(velo_d[1])} {torch.max(velo_d[1])} {torch.min(velo_d[1])}")

            M_VELOD = 14.0833
            S_VELOD = 8.7353
            velo_d = (velo_d * S_VELOD)+ M_VELOD
            cam2_tensor = np.transpose(cam2, (2, 0, 1)) 
            cam2_tensor = torch.Tensor(cam2_tensor)
            cam2_tensor = cam2_tensor.unsqueeze(0)
            cam2_tensor = crop_img(cam2_tensor, 352, 1216)
            batch_data = {"rgb": cam2_tensor, "d": velo_d,\
              'position': args.position, 'K': args.K}
            batch_data = {
                key: val.to(DEVICE)
                for key, val in batch_data.items() if val is not None
            }
            pred_d = model_d(batch_data)
            # depth_path = os.path.join(args.data_folder_save, str(i).zfill(10) + '_output.png')
            # rgb_path = os.path.join(args.data_folder_save, str(i).zfill(10) + '_output_rgb.png')
            # print(depth_path)
            # vis_utils.save_depth_as_uint16png_upload(pred_d, depth_path)
            # im = Image.fromarray(cam2)
            # im.save(rgb_path)
            # forward -------------------------- CUR
            # pred_r, pred_t = model(velo_d, cam2_d)

            # M_VELOD = 14.0833
            # S_VELOD = 8.7353
            # pred_d = (pred_d - M_VELOD) / S_VELOD
            pred_d = (pred_d - M_VELOD) / S_VELOD
            pred_d = 30-pred_d
            pred_r, pred_t = model(pred_d, cam2_d)
            
            loss = loss_fn((pred_r, pred_t), torch.split(gt_err_norm, 3, dim=1), velo, T)

            # collect statistics
            running_loss += loss.item()
            running_mean = running_loss / (i + 1)
            pbar.set_description("Epoch %d, valid running loss: %.4f" % (epoch + 1, running_mean))

            gt_splits = np.split(gt_err, 2, axis=1)
            pred_r, pred_t = loader.dataset.destandardize(pred_r.detach().cpu().numpy(), pred_t.detach().cpu().numpy())
            rot_err += np.abs(pred_r-gt_splits[0]).mean(axis=0)
            trans_err += np.abs(pred_t-gt_splits[1]).mean(axis=0)

        # write results
        rot_err = rot_err / (i + 1)
        trans_err = trans_err / (i + 1)
        logger.add_scalar('Valid/Loss/loss', running_mean, global_i)
        logger.add_scalar('Valid/Loss/roll', rot_err[0], global_i)
        logger.add_scalar('Valid/Loss/pitch', rot_err[1], global_i)
        logger.add_scalar('Valid/Loss/yaw', rot_err[2], global_i)
        logger.add_scalar('Valid/Loss/x', trans_err[0], global_i)
        logger.add_scalar('Valid/Loss/y', trans_err[1], global_i)
        logger.add_scalar('Valid/Loss/z', trans_err[2], global_i)
    return running_mean


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/home/username/dataset/KITTI/', help='KITTI dataset root directory.')
    parser.add_argument('--batch', type=int, default=1, help='Batch size.')
    parser.add_argument('--ckpt', type=str, help='Path to the saved model in saved folder.')
    parser.add_argument('--ckpt_no_lr', action='store_true', help='Ignore lr in the checkpoint.')
    parser.add_argument('--model', type=int, default=1, help='Select model variant to test.')
    parser.add_argument('--rotation_offsest', type=float, default=10.0, help='Random rotation error range.')
    parser.add_argument('--translation_offsest', type=float, default=0.2, help='Random translation error range.')
    parser.add_argument('--epoch', type=int, default=50, help='Epochs to train.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Model learning rate.')
    parser.add_argument('--patience', type=int, default=6, help='Patience for reducing lr.')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='Factor for reducing lr.')
    parser.add_argument('--loss_a', type=float, default=1.0, help='Loss factor for rotation & translation errors.')
    parser.add_argument('--loss_b', type=float, default=1.0, help='Loss factor for point cloud center errors.')
    parser.add_argument('--loss_c', type=float, default=1.0, help='Loss factor for point cloud errors.')
    parser.add_argument('--exp_name', type=str, default=f'exp_{datetime.now().strftime("%H%M_%d%m%Y")}',
                        help='Loss factor for translation errors.')
    parser.add_argument('--stat', action='store_true', help='Calculate dataset statistics.')
    ##################################
    parser.add_argument('-n',
                    '--network-model',
                    type=str,
                    default="e",
                    choices=["e", "pe"],
                    help='choose a model: enet or penet'
                    )
    parser.add_argument('--workers',
                        default=4,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs',
                        default=100,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run (default: 100)')
    parser.add_argument('--start-epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--start-epoch-bias',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number bias(useful on restarts)')
    parser.add_argument('-c',
                        '--criterion',
                        metavar='LOSS',
                        default='l2',
                        choices=criteria_d.loss_names,
                        help='loss function: | '.join(criteria_d.loss_names) +
                        ' (default: l2)')
    parser.add_argument('-b',
                        '--batch-size',
                        default=1,
                        type=int,
                        help='mini-batch size (default: 1)')
    # parser.add_argument('--lr',
    #                     '--learning-rate',
    #                     default=1e-3,
    #                     type=float,
    #                     metavar='LR',
    #                     help='initial learning rate (default 1e-5)')
    parser.add_argument('--weight-decay',
                        '--wd',
                        default=1e-6,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 0)')
    parser.add_argument('--print-freq',
                        '-p',
                        default=10,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--resume',
                        default='',
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--data-folder',
                        default='./dataset/kitti_depth/depth',
                        type=str,
                        metavar='PATH',
                        help='data folder (default: none)')
    parser.add_argument('--data-folder-rgb',
                        default='./dataset/kitti_raw',
                        type=str,
                        metavar='PATH',
                        help='data folder rgb (default: none)')
    parser.add_argument('--data-folder-save',
                        default='./depth/result_raw',
                        type=str,
                        metavar='PATH',
                        help='data folder test results(default: none)')
    parser.add_argument('-i',
                        '--input',
                        type=str,
                        default='rgbd',
                        choices=input_options,
                        help='input: | '.join(input_options))
    parser.add_argument('--val',
                        type=str,
                        default="select",
                        choices=["select", "full"],
                        help='full or select validation set')
    parser.add_argument('--jitter',
                        type=float,
                        default=0.1,
                        help='color jitter for images')
    # parser.add_argument('--rank-metric',
    #                     type=str,
    #                     default='rmse',
    #                     choices=[m for m in dir(Result()) if not m.startswith('_')],
    #                     help='metrics for which best result is saved')

    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')
    parser.add_argument('-f', '--freeze-backbone', action="store_true", default=False,
                        help='freeze parameters in backbone')
    parser.add_argument('--test', action="store_true", default=False,
                        help='save result kitti test dataset for submission')
    parser.add_argument('--cpu', action="store_true", default=False, help='run on cpu')

    #random cropping
    parser.add_argument('--not-random-crop', action="store_true", default=False,
                        help='prohibit random cropping')
    parser.add_argument('-he', '--random-crop-height', default=320, type=int, metavar='N',
                        help='random crop height')
    parser.add_argument('-w', '--random-crop-width', default=1216, type=int, metavar='N',
                        help='random crop height')

    #geometric encoding
    parser.add_argument('-co', '--convolutional-layer-encoding', default="xyz", type=str,
                        choices=["std", "z", "uv", "xyz"],
                        help='information concatenated in encoder convolutional layers')

    #dilated rate of DA-CSPN++
    parser.add_argument('-d', '--dilation-rate', default="2", type=int,
                        choices=[1, 2, 4],
                        help='CSPN++ dilation rate')
       
    ##################################
    return parser.parse_args()


if __name__ == '__main__':
    # parse args
    args = arg_parser()
    print(args)

    # calculating dataset statistics
    if args.stat:
        train_ds = CalibDataset(path=args.dataset, mode='train')
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
        run_stat(train_loader)
        exit(0)

    # creating dirs
    result_base = Path('.').absolute().parent.joinpath('results').joinpath(args.exp_name)
    log_dir = result_base.joinpath('log')
    ckpt_dir = result_base.joinpath('ckpt')
    mod_dir = result_base.joinpath('mod')
    result_base.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    mod_dir.mkdir(exist_ok=True)
    # save critical files
    shutil.copy2('train_with_dense.py', mod_dir.as_posix())
    shutil.copy2('train_dense.sh', mod_dir.as_posix())
    shutil.copy2('./net/CalibrationNet.py', mod_dir.as_posix())
    shutil.copy2('./net/Convolution.py', mod_dir.as_posix())
    shutil.copy2('./net/SpatialPyramidPooling.py', mod_dir.as_posix())

    # build model
    model = get_model(args.model)
    print(f"Model trainable parameters: {count_parameters(model)}")

    # build optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # scheduler
    scheduler = ReduceLROnPlateau(optimizer, factor=args.lr_factor, patience=args.patience,
                                  verbose=True, min_lr=1e-8, cooldown=2)

    # move model to gpu
    model.to(DEVICE)

    # load ckpt
    if args.ckpt:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])
        scheduler.load_state_dict(ckpt['scheduler'])

        trained_epochs = ckpt['epoch']
        global_i = ckpt['global_i']
        best_val = ckpt['best_val']
        train_loss = ckpt['loss']
        rng = ckpt['rng']
        torch.set_rng_state(rng)

        if args.ckpt_no_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
    else:
        trained_epochs = 0
        global_i = 0
        best_val = 9999
        train_loss = 9999
    valid_loss = 9999

    # create data loaders
    train_ds = CalibDataset(path=args.dataset, mode='train',
                            rotation_offset=args.rotation_offsest, translation_offset=args.translation_offsest)
    print("HERE!!")
    val_ds = CalibDataset(path=args.dataset, mode='val',
                          rotation_offset=args.rotation_offsest, translation_offset=args.translation_offsest)
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=True, num_workers=0,
                            pin_memory=True, drop_last=True)

    # loss
    criteria = Loss(dataset=train_ds, alpha=args.loss_a, beta=args.loss_b, gamma=args.loss_c)

    # summary writer
    writer = SummaryWriter(log_dir=log_dir.as_posix(), purge_step=global_i)

    ##################################
    args.result = os.path.join('..', 'results')
    args.use_rgb = ('rgb' in args.input)
    args.use_d = 'd' in args.input
    args.use_g = 'g' in args.input
    args.val_h = 352
    args.val_w = 1216
    args.position = None
    args.K=None
    print(args)

    cuda = torch.cuda.is_available() and not args.cpu
    if cuda:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("=> using '{}' for computation.".format(DEVICE))

    # define loss functions
    depth_criterion = criteria_d.MaskedMSELoss() if (
        args.criterion == 'l2') else criteria_d.MaskedL1Loss()

    #multi batch
    multi_batch_size = 1

    position_np = CoordConv.AddCoordsNp(args.val_h, args.val_w)
    position_np = position_np.call()
    K_np = load_calib()


    position_np = np.transpose(position_np, (2, 0, 1)) 
    args.position = torch.Tensor(position_np)
    args.position = args.position.unsqueeze(0)

    args.K = torch.Tensor(K_np)
    args.K = args.K.unsqueeze(0)

    ############################# 
    ##################
    checkpoint = None
    is_eval = False
    if args.evaluate:                
        args_new = args
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}' ... ".format(args.evaluate),
                end='')
            checkpoint = torch.load(args.evaluate, map_location=DEVICE)                                    
            #args = checkpoint['args']
            args.start_epoch = checkpoint['epoch'] + 1
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            is_eval = True
            print("Completed.")
        else:
            is_eval = True
            print("No model found at '{}'".format(args.evaluate))
    elif args.resume:  # optionally resume from a checkpoint        
        args_new = args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}' ... ".format(args.resume),
                end='')
            checkpoint = torch.load(args.resume, map_location=DEVICE)
            args.start_epoch = checkpoint['epoch'] + 1
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            print("Completed. Resuming from epoch {}.".format(
                checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            # return
    
    #choose model
    print("=> creating model and optimizer ... ", end='')
    model_d = None
    penet_accelerated = False
    if (args.network_model == 'e'):
        model_d = ENet(args).to(DEVICE)
    elif (is_eval == False):
        if (args.dilation_rate == 1):
            model_d = PENet_C1_train(args).to(DEVICE)
        elif (args.dilation_rate == 2):
            model_d = PENet_C2_train(args).to(DEVICE)
        elif (args.dilation_rate == 4):
            model_d = PENet_C4(args).to(DEVICE)
            penet_accelerated = True
    else:
        if (args.dilation_rate == 1):
            model_d = PENet_C1(args).to(DEVICE)
            penet_accelerated = True
        elif (args.dilation_rate == 2):
            model_d = PENet_C2(args).to(DEVICE)
            penet_accelerated = True
        elif (args.dilation_rate == 4):
            model_d = PENet_C4(args).to(DEVICE)
            penet_accelerated = True
    if (penet_accelerated == True):
        model_d.encoder3.requires_grad = False
        model_d.encoder5.requires_grad = False
        model_d.encoder7.requires_grad = False

    model_d_named_params = None
    model_d_bone_params = None
    model_d_new_params = None
    
    # model_pre = get_model(args.model)
    # model_pre.to(DEVICE)
    # ckpt = torch.load("/workspace/src/results/Deep_F/ckpt/Epoch29_val_0.8500.tar")

    # model_pre.load_state_dict(ckpt['model'])
    if checkpoint is not None:
        #print(checkpoint.keys())
        if (args.freeze_backbone == True):
            model_d.backbone.load_state_dict(checkpoint['model'])
        else:
            model_d.load_state_dict(checkpoint['model'], strict=False)
        #optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> checkpoint state loaded.")
    
    os.makedirs(args.data_folder_save, exist_ok=True)
    
    logger = helper.logger(args)
    if checkpoint is not None:
        logger.best_result = checkpoint['best_result']
        del checkpoint
    print("=> logger created.")
    ##################


    # train model
    for epoch in range(trained_epochs, trained_epochs+args.epoch):
        # train
        print("Train len {}".format(len(train_ds)))
        print("Train len {}".format(len(train_loader)))
        # train_loss, global_i = train_one_epoch(None, model, criteria, optimizer, train_loader, epoch, global_i, writer)
        train_loss, global_i = train_one_epoch(model_d, model, criteria, optimizer, train_loader, epoch, global_i, writer)

        # valid
        valid_loss = validation(model_d, model, criteria, val_loader, epoch, global_i, writer)

        # if train_loss < best_val:
        if True:
        # if valid_loss < best_val:
            # print(f'Best model (val:{train_loss:.04f}) saved.')
            print(f'2Best model (val:{valid_loss:.04f}) saved.')
            best_val = train_loss
            # best_val = valid_loss
            torch.save({
                'epoch': epoch+1,
                'global_i': global_i,
                'best_val': best_val,
                'loss': train_loss,
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'rng': torch.get_rng_state()
            }, ckpt_dir.joinpath(f'Epoch{epoch+1}_val_{best_val:.04f}.tar').as_posix())

        # update scheduler
        # scheduler.step(train_loss)
        scheduler.step(valid_loss)

    # final model
    torch.save({
        'epoch': trained_epochs+args.epoch,
        'global_i': global_i,
        'best_val': best_val,
        'loss': train_loss,
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'rng': torch.get_rng_state()
    }, ckpt_dir.joinpath(f'Epoch{trained_epochs+args.epoch}_val_{valid_loss:.04f}.tar').as_posix())
