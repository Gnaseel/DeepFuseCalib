import random
import numpy as np
import argparse
import cv2
import os
import time
import torch
from torch import Tensor
import torch.utils.data
from torch.nn import Module, MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Optional, List
from typing_extensions import Literal
from net.CalibrationNet import get_model
from dataset import EvalDataset as CalibDataset
from utils import count_parameters, inv_transform_vectorized, phi_to_transformation_matrix_vectorized, \
    lidar_projection, inv_transform, phi_to_transformation_matrix, merge_color_img_with_depth, calib_np_to_phi

###################
from depth.dataloaders.kitti_loader import load_calib, input_options, KittiDepth
# from depth.metrics_d import AverageMeter, Result
from depth import criteria_d
from depth import helper
from depth import vis_utils

from depth.model import ENet
from depth.model import PENet_C1_train
from depth.model import PENet_C2_train
#from model import PENet_C4_train (Not Implemented)
from depth.model import PENet_C1
from depth.model import PENet_C2
from depth.model import PENet_C4

from depth import CoordConv

from depth.dataloaders import transforms
from torchvision.utils import save_image
from PIL import Image
import sys
###################


# set seeds
SEED = 53
rng = np.random.default_rng(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = 'cuda:2'





class Loss(Module):
    def __init__(self, dataset: CalibDataset, reduction: str = 'mean',
                 alpha: float = 0.5, beta: float = 1, gamma: float = 0.5, cache: int = 60000):
        super(Loss, self).__init__()
        self.dataset = dataset
        self.cache = cache

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
            # project
            pts3d_cam = T_.mm(pts3d.t()).t()[:self.cache, :].unsqueeze(0)
            pts3d_cam_recalib = T_recalib_.mm(pts3d.t()).t()[:self.cache, :].unsqueeze(0)
            pts3d_cam_l.append(pts3d_cam)
            pts3d_cam_recalib_l.append(pts3d_cam_recalib)
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
    # print(f' {height} {width} {h} {w}')
    # print(f'Margin {top_margin} {left_margin}')
    if len(img.shape) == 3:
        image = img[top_margin:top_margin + h, left_margin:left_margin + w, :]
    else:
        image = img[:,:,top_margin:top_margin + h, left_margin:left_margin + w]
    return image
def run_visualization(model_d, model: Module, loader: DataLoader, loss_fn: Loss, fout: Optional[str] = None, downscale: int = 2) -> None:
    h, w = 375//downscale, 1242//downscale

    save_path = '/workspace/data2/result_calibration_img/deep_viz'

    # cv2.namedWindow('Visualization', flags=cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Visualization', w, h*3)

    running_loss = 0
    running_mean = 0
    running_time = 0
    running_msee = 0
    rot_err = np.zeros((len(loader.dataset), 6), np.float32)
    trans_err = np.zeros((len(loader.dataset), 6), np.float32)
    pbar = tqdm(loader)
    model.eval()

    with torch.no_grad():
        i = 0
        for i, sample in enumerate(pbar):
            cam2_d, velo_d, gt_err, gt_err_norm, cam2, velo, T, P = sample[0].to(DEVICE), sample[1].to(DEVICE), sample[2], sample[3].to(DEVICE), \
                                                              sample[4].to(DEVICE), sample[5].to(DEVICE), sample[6].to(DEVICE), sample[7].to(DEVICE)

            # SHAPE CHANGE
            # print(f'SHAPE PRE  {cam2_d.shape}')
            crop_start = time.time()        #-----------------------------------
            cam2_d = crop_img(cam2_d, 352, 1216)
            crop_end = time.time()          #-----------------------------------
            # print(f'SHAPE POST {cam2_d.shape}')
            pred_r, pred_t = model(velo_d, cam2_d)
            model_end = time.time()         #-----------------------------------
            # print(f"TIME 0------------------- {t2-t}")
            running_time += (time.time() - crop_end)
            loss = loss_fn((pred_r, pred_t), torch.split(gt_err_norm, 3, dim=1), velo, T)
            loss_end = time.time()          #-----------------------------------

            # collect statistics
            running_loss += loss.item()
            running_mean = running_loss / (i + 1)
            pbar.set_description("Valid running loss: %.4f" % (running_mean))
            # print(f" GT Shape {gt_err.shape} {gt_err}")

            gt_splits = np.split(gt_err.numpy(), 2, axis=1)
            pred_r, pred_t = loader.dataset.destandardize(pred_r.detach().cpu().numpy(), pred_t.detach().cpu().numpy())
            rot_err[i, :3] = np.abs(pred_r-gt_splits[0]).mean(axis=0)
            trans_err[i, :3] = np.abs(pred_t-gt_splits[1]).mean(axis=0)
            save_end = time.time()          #-----------------------------------

            # lidar projections
            cam2 = cam2[0].cpu().numpy()
            velo = velo[0].cpu().numpy()
            T = T[0].cpu().numpy()
            P = P[0].cpu().numpy()
            gt_err = gt_err.numpy()[0]

            rot_err[i, 3:] = gt_err[:3]
            trans_err[i, 3:] = gt_err[3:]

            T_err = phi_to_transformation_matrix(gt_err)
            T_composed = T_err @ T
            uncalib = lidar_projection(velo, T_composed, P, cam2.shape[:2], downscale=downscale)

            T_est = phi_to_transformation_matrix(np.concatenate([pred_r[0], pred_t[0]]))
            T_patch = inv_transform(T_est)
            T_recalib = T_patch @ T_composed
            recalib = lidar_projection(velo, T_recalib, P, cam2.shape[:2], downscale=downscale)
            calibgt = lidar_projection(velo, T, P, cam2.shape[:2], downscale=downscale)

            proj_end = time.time()          #-----------------------------------

            print(f"Crop  Time {crop_end-crop_start}")
            print(f"Model Time {model_end-crop_end}")
            print(f"Loss  Time {loss_end - model_end}")
            print(f"Save  Time {save_end - loss_end}")
            print(f"Proj  Time {proj_end-save_end}")


            # calculate running msee
            running_msee += np.linalg.norm((T_est - T_err))

            # show img
            if downscale != 1:
                cam2 = cv2.resize(cam2, (w, h), interpolation=cv2.INTER_AREA)
            uncalib = cv2.putText(merge_color_img_with_depth(cv2.cvtColor(cam2, cv2.COLOR_RGB2BGR), uncalib),
                                  'Input', (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            recalib = cv2.putText(merge_color_img_with_depth(cv2.cvtColor(cam2, cv2.COLOR_RGB2BGR), recalib),
                                  'Pred', (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            calibgt = cv2.putText(merge_color_img_with_depth(cv2.cvtColor(cam2, cv2.COLOR_RGB2BGR), calibgt),
                                  'GT', (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            img = np.concatenate([uncalib, recalib, calibgt], axis=0)

            # cv2.imshow('Visualization', img)
            # One model

            os.makedirs(save_path,exist_ok=True)
            # print(os.path.join(save_path,'Visualization_'+str(i)+'.png'))
            cv2.imwrite(os.path.join(save_path,'Visualization_'+str(i)+'.png'), img)
            if fout is not None:
            #     writer.write(img)
                cv2.imwrite(Path(fout).joinpath('img').joinpath(f'{i:04d}.png').as_posix(), img)
            key = cv2.waitKey(int(1 / 60 * 1000))
            if (key & 0xFF == ord('q')) or (key & 0xFF == 27):
                break
            # t3 = time.time()
            # print(f"TIME 1------------------- {t3-t}")

    # if fout is not None:
    #     writer.release()
    # show results
    avg_inference_time = running_time / (i + 1)
    print(f'Running loss: {running_mean:.04f}')
    print(f'MSEE: {running_msee/(i + 1):.04f}')
    print(f'Avg. inference speed: {avg_inference_time:.04f} s')
    print(f'Rotation Total Shape {np.rad2deg(rot_err).shape}')
    print(f'Rotation Total Shape {rot_err}')
    print(f'Rotation mean errors (degree): {np.rad2deg(rot_err.mean(axis=0)[:3])}')
    print(f'Rotation max errors (degree): {np.rad2deg(rot_err.max(axis=0)[:3])}')
    print(f'Rotation min errors (degree): {np.rad2deg(rot_err.min(axis=0)[:3])}')
    print(f'Rotation std errors (degree): {np.rad2deg(rot_err.std(axis=0)[:3])}')
    print(f'Translation mean errors (meter): {trans_err.mean(axis=0)[:3]}')
    print(f'Translation max errors (meter): {trans_err.max(axis=0)[:3]}')
    print(f'Translation min errors (meter): {trans_err.min(axis=0)[:3]}')
    print(f'Translation std errors (meter): {trans_err.std(axis=0)[:3]}')

    # if fout is not None:
    #     pd.DataFrame(rot_err).to_csv(Path(fout).joinpath('rot_stat.csv'), header=['row', 'pitch', 'yaw', 'row_gt', 'pitch_gt', 'yaw_gt'], index=False)
    #     pd.DataFrame(trans_err).to_csv(Path(fout).joinpath('trans_stat.csv'), header=['x', 'y', 'z', 'x_gt', 'y_gt', 'z_gt'], index=False)
    return


def run_iterative(model_d, models: List[Module], loader: DataLoader, loss_fn: Loss, offsets: List[Tuple[float, float]], fout: Optional[str] = None, downscale: int = 2) -> None:
    h, w = 375//downscale, 1242//downscale
    # h, w = 352//downscale, 1216//downscale
    
    # ITER
    save_path = '/workspace/data2/result_calibration_img/deep_it'

    # if fout is not None:
    #     pout = Path(fout).joinpath('model_inference.avi')
    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     writer = cv2.VideoWriter(pout.as_posix(), fourcc, 10.0, (1242, 375*3))

    # cv2.namedWindow('Visualization', flags=cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Visualization', 1250, 380*3)

    running_loss = 0
    running_mean = 0
    running_time = 0
    running_msee = 0
    rot_err = np.zeros((len(loader.dataset), 3), np.float32)
    trans_err = np.zeros((len(loader.dataset), 3), np.float32)
    pbar = tqdm(loader)
    
    for m in models:
        m.eval()

    with torch.no_grad():
        i = 0
        for i, sample in enumerate(pbar):

            crop_start = time.time()        #-----------------------------------
            cam2_d, velo_d, gt_err, gt_err_norm, cam2, velo, T, P = sample[0].to(DEVICE), sample[1].to(DEVICE), sample[2], sample[3].to(DEVICE), \
                                                              sample[4].to(DEVICE), sample[5].to(DEVICE), sample[6].to(DEVICE), sample[7].to(DEVICE)

            print(f"TYPE----------------- {type(velo_d)}")
            print(f"TYPE----------------- {type(velo_d)}")
            print(f"TYPE----------------- {type(velo_d)}")
            print(f" EVAL Data {velo_d.shape}  {torch.mean(velo_d)} {torch.max(velo_d)} {torch.min(velo_d)}")

            cam2_d = crop_img(cam2_d, 352, 1216)

            # prepare matrix            
            cam2 = cam2[0].cpu().numpy()            
            velo_np = velo[0].cpu().numpy()
            T_np = T[0].cpu().numpy()
            P = P[0].cpu().numpy()
            gt_err = gt_err.numpy()[0]
            T_err = phi_to_transformation_matrix(gt_err)
            T_composed = T_err @ T_np
            crop_end = time.time()          #-----------------------------------

            # timer start
            t = time.time()

            
            # run iterative mode
            # Final R matrix with start from GT (destandarized)
            phi = T_composed.copy()
            # Final R matrix (destandarized)
            T_est_final = np.eye(4, dtype=np.float32)
            loss_phi = np.eye(4, dtype=np.float32)

            idx=0
            model_time=0
            proj_time=0
            velo_d_raw=None
            for m, (rot_offset, trans_offset) in zip(models, offsets):

                # Model Input
                model_start = time.time()
                pred_r, pred_t = m(velo_d, cam2_d)
                model_end = time.time()
                model_time += model_end - model_start


                pred_r_real, pred_t_real = loader.dataset.destandardize(pred_r.detach().cpu().numpy(), pred_t.detach().cpu().numpy(),
                                                                        rot_offset=rot_offset, trans_offset=trans_offset)
                T_est = phi_to_transformation_matrix(np.concatenate([pred_r_real[0], pred_t_real[0]]))
                T_est_final = T_est @ T_est_final
                T_patch = inv_transform(T_est)
                phi = T_patch @ phi
                loss_phi = phi_to_transformation_matrix(np.concatenate([pred_r[0].detach().cpu().numpy(), pred_t[0].detach().cpu().numpy()])) @ loss_phi

                post_end = time.time()
                # phi = phi @ T_patch 
                # now = T_patch @ T_composed
    
                # print(save_path)
                velo_d_np = lidar_projection(velo_np, phi, P, cam2.shape[:2], crop=cam2_d.shape[2:])
                proj_end = time.time()
                proj_time +=proj_end - post_end
                # Image Save Mode
                # velo_d_np_downed = lidar_projection(velo_np, phi, P, cam2.shape[:2], downscale=downscale)
                # if downscale != 1: # Save mode
                #     cam2_downed = cv2.resize(cam2, (w, h), interpolation=cv2.INTER_AREA)
                # imim = merge_color_img_with_depth(cv2.cvtColor(cam2_downed, cv2.COLOR_RGB2BGR), velo_d_np_downed)
                # cv2.imwrite(os.path.join(save_path,'Visualization_'+str(i)+'_'+str(idx)+'.png'), imim)

                

                velo_d = torch.from_numpy(velo_d_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
                ###################
                velo_d_raw = velo_d
                break
                ###################
                M_VELOD = 14.0833
                S_VELOD = 8.7353
                velo_d = (velo_d - M_VELOD) / S_VELOD

                what_end = time.time()


                idx +=1



            model_d.eval()
            lr = 0
            
            cam2_tensor = np.transpose(cam2, (2, 0, 1)) 
            cam2_tensor = torch.Tensor(cam2_tensor)
            cam2_tensor = cam2_tensor.unsqueeze(0)
            cam2_tensor = crop_img(cam2_tensor, 352, 1216)

            batch_data = {"rgb": cam2_tensor, "d": velo_d_raw,\
                      'position': args.position, 'K': args.K}
            batch_data = {
                key: val.to(DEVICE)
                for key, val in batch_data.items() if val is not None
            }

            pred_d = model_d(batch_data)
            print(batch_data['K'])
            print(f"Shape of pred {pred_d.shape}")
            # print(f"Data {velo_d_raw.shape}  {torch.mean(velo_d_raw)} {torch.max(velo_d_raw)}")
            print(f"Data {velo_d_raw.shape}  {torch.mean(velo_d_raw)} {torch.max(velo_d_raw)} {torch.min(velo_d_raw)}")
            print(f"Data {velo_d_raw.shape}  {torch.mean(velo_d_raw)} {torch.max(velo_d_raw)} {torch.min(velo_d_raw)}")

            str_i = str(i)
            path_i = str_i.zfill(10) + '_output.png'
            path = os.path.join(args.data_folder_save, path_i)
            rgb_path = os.path.join(args.data_folder_save, str(i).zfill(10) + '_output_rgb.png')
            im = Image.fromarray(cam2)
            im.save(rgb_path)
            # print("pred")
            # print(type(pred_d))            
            # pred_d = torch.tensor(pred_d)
            # print(type(pred_d))

            cam2_d = pred_d 
            vis_utils.save_depth_as_uint16png_upload(pred_d, path)
            print(f"Path {path}")

            # sys.exit()
            ###############
            T_recalib = phi.copy()


            # new_r, new_t = loader.dataset.destandardize(pred_r.detach().cpu().numpy(), pred_t.detach().cpu().numpy())

            new_vec = calib_np_to_phi(T_est_final)
            # print(f"NEW vec {new_vec}")
            loss_vec = calib_np_to_phi(loss_phi)
            new_r = new_vec[:3]
            new_t = new_vec[3:]
            loss_r = loss_vec[:3]
            loss_t = loss_vec[3:]

            # timer stop
            running_time += (time.time() - t)
            # loss = loss_fn((pred_r, pred_t), torch.split(gt_err_norm, 3, dim=1), velo, T)
            loss = loss_fn((torch.from_numpy(loss_r.reshape(1,3)).to(DEVICE), torch.from_numpy(loss_t.reshape(1,3)).to(DEVICE)), torch.split(gt_err_norm, 3, dim=1), velo, T)

            # collect statistics
            running_loss += loss.item()
            running_mean = running_loss / (i + 1)
            pbar.set_description("Valid running loss: %.4f" % (running_mean))

            running_msee += np.linalg.norm((T_est_final - T_err))

            # print(f"Shape {pred_r_real.shape} {pred_t_real.shape}")
            # print(f" GT Shape {gt_err.shape} {gt_err}")

            gt_splits = np.split(gt_err, 2, axis=0)
            rot_err[i] = np.abs(new_r-gt_splits[0])
            trans_err[i] = np.abs(new_t-gt_splits[1])

            continue
            # print(f"Prep  Time {crop_end-crop_start}")
            # print(f"Model Time {model_time}")
            # print(f"Proj  Time {proj_time}")
            # print(f"Proj2 Time {what_end - proj_end}")
            # print(f"Proj  Time {proj_end-save_end}")
            # print(f"PRED _FIRST {new_r}   {new_t}" )
            # print(f"GT   _FIRST {gt_splits[0]}   {gt_splits[1]}" )
            # print(f"ERROR {rot_err[i]}   {trans_err[i]}" )
            # print(f'Rotation Total Shape {np.rad2deg(rot_err).shape}')
            # print(f'Rotation mean errors (degree): {np.rad2deg(rot_err.mean(axis=0))}')
            # print(f'Translation mean errors (meter): {trans_err.mean(axis=0)}')

            vis_start=time.time()
            # lidar projections -- raw errors
            uncalib = lidar_projection(velo_np, T_composed, P, cam2.shape[:2], downscale=downscale)
            # lidar projections -- model predictions
            recalib = lidar_projection(velo_np, T_recalib, P, cam2.shape[:2], downscale=downscale)
            # lidar projections -- ground truth
            calibgt = lidar_projection(velo_np, T_np, P, cam2.shape[:2], downscale=downscale)
            vis_end=time.time()
            print(f"Vis  Time {vis_end - vis_start}")

            # calculate running msee

            # show img

            if downscale != 1:
                cam2 = cv2.resize(cam2, (w, h), interpolation=cv2.INTER_AREA)
            uncalib = cv2.putText(merge_color_img_with_depth(cv2.cvtColor(cam2, cv2.COLOR_RGB2BGR), uncalib),
                                  'Input', (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            recalib = cv2.putText(merge_color_img_with_depth(cv2.cvtColor(cam2, cv2.COLOR_RGB2BGR), recalib),
                                  'Pred', (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            calibgt = cv2.putText(merge_color_img_with_depth(cv2.cvtColor(cam2, cv2.COLOR_RGB2BGR), calibgt),
                                  'GT', (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            img = np.concatenate([uncalib, recalib, calibgt], axis=0)
            # cv2.imshow('Visualization', img)
            # Iter
            # print(f"SAVE PATH {save_path}")
            os.makedirs(save_path,exist_ok=True)

            cv2.imwrite(os.path.join(save_path,'Visualization_'+str(i)+'.png'), img)
            if fout is not None:
                writer.write(img)
            key = cv2.waitKey(int(1 / 60 * 1000))
            if (key & 0xFF == ord('q')) or (key & 0xFF == 27):
                break

    if fout is not None:
        writer.release()
    # show results
    avg_inference_time = running_time / (i + 1)
    print(f'Running loss: {running_mean:.04f}')
    print(f'MSEE: {running_msee/(i + 1):.04f}')
    print(f'Avg. inference speed: {avg_inference_time:.04f} s')
    print(f'Rotation Total Shape {np.rad2deg(rot_err).shape}')
    print(f'Rotation Total (degree): {rot_err}')
    print(f'Rotation mean errors (degree): {np.rad2deg(rot_err.mean(axis=0))}')
    print(f'Rotation max errors (degree): {np.rad2deg(rot_err.max(axis=0))}')
    print(f'Rotation min errors (degree): {np.rad2deg(rot_err.min(axis=0))}')
    print(f'Rotation std errors (degree): {np.rad2deg(rot_err.std(axis=0))}')
    print(f'Translation mean errors (meter): {trans_err.mean(axis=0)}')
    print(f'Translation max errors (meter): {trans_err.max(axis=0)}')
    print(f'Translation min errors (meter): {trans_err.min(axis=0)}')
    print(f'Translation std errors (meter): {trans_err.std(axis=0)}')
    return


def arg_parser():
    parser = argparse.ArgumentParser()
    # common args
    parser.add_argument('--dataset', type=str, default='/workspace/data/KITTI/', help='Model learning rate.')
    parser.add_argument('--model', type=int, default=1, help='Select model variant to test.')
    parser.add_argument('--loss_a', type=float, default=1.0, help='Loss factor for rotation & translation errors.')
    parser.add_argument('--loss_b', type=float, default=2.0, help='Loss factor for point cloud center errors.')
    parser.add_argument('--loss_c', type=float, default=2.0, help='Loss factor for point cloud errors.')

    # run visualization task
    parser.add_argument('--visualization', action='store_true', help='Show online running test.')
    parser.add_argument('--ckpt', type=str, help='Path to the saved model in saved folder.')
    parser.add_argument('--rotation_offsest', type=float, default=10.0, help='Random rotation error range.')
    parser.add_argument('--translation_offsest', type=float, default=0.2, help='Random translation error range.')

    # run in iterative mode
    parser.add_argument('--iterative', action='store_true', help='Show online iterative running test.')
    parser.add_argument('--ckpt_list', type=str, help='One or more paths to the saved model in saved folder. The first one will be used first.', nargs='+')
    # parser.add_argument('--rotation_offsests', type=float, default=[10.0, 2.0], help='List of random rotation error range.', nargs='+')
    # parser.add_argument('--translation_offsests', type=float, default=[0.2, 0.2], help='List of random translation error range.', nargs='+')
    parser.add_argument('--rotation_offsests', type=float, default=[10.0, 2.0, 2.0, 2.0], help='List of random rotation error range.', nargs='+')
    parser.add_argument('--translation_offsests', type=float, default=[0.2, 0.2, 0.2, 0.2], help='List of random translation error range.', nargs='+')

    # output path
    parser.add_argument('--out_path', type=str, help='Path to store the visualized video.')


    # python eval.py --ckpt ../results/HOPE2/ckpt/Epoch47_val_0.0475.tar --visualization --rotation_offsest 10 --translation_offsest 0.2
    # python eval.py --ckpt_list ../results/HOPE-10-0.2/ckpt/Epoch65_val_0.2983.tar ../results/HOPE2/ckpt/Epoch47_val_0.0475.tar --iterative --rotation_offsest 10 --translation_offsest 0.2

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
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=1e-3,
                        type=float,
                        metavar='LR',
                        help='initial learning rate (default 1e-5)')
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

    # print("position_np")
    # print(position_np)
    # print("K_np")
    # print(K_np)
    # print(position_np.shape)
    # print(K_np.shape)


    # args.position = torch.from_numpy(position_np)
    # args.K=torch.from_numpy(K_np)
    # args.position = position_np
    # args.K=K_np

    position_np = np.transpose(position_np, (2, 0, 1)) 
    args.position = torch.Tensor(position_np)
    args.position = args.position.unsqueeze(0)

    args.K = torch.Tensor(K_np)
    args.K = args.K.unsqueeze(0)

    # position_np = position_np.reshape((1,2,352,1216))
    # K_np = K_np.reshape((1,3,3))
    # args.position = torch.from_numpy(position_np)
    # args.K=torch.from_numpy(K_np)

    # print(args.position)
    # sys.exit()

    # position=position.view([2,352,1216])
    # args.position=position.unsqueeze(0)
    # args.K=K.unsqueeze(0)


    # print("tensor view")
    # print(position.shape)
    # print(K.shape)
    # sys.exit()
    #############################    
    


    # dataset & loss    
    test_ds = CalibDataset(path=args.dataset, mode='test', rotation_offset=args.rotation_offsest, translation_offset=args.translation_offsest)
    criteria = Loss(dataset=test_ds, alpha=args.loss_a, beta=args.loss_b, gamma=args.loss_c)

    ###################
    # checkpoint        
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
    if checkpoint is not None:
        #print(checkpoint.keys())
        if (args.freeze_backbone == True):
            model_d.backbone.load_state_dict(checkpoint['model'])
        else:
            model_d.load_state_dict(checkpoint['model'], strict=False)
        #optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> checkpoint state loaded.")
    
    logger = helper.logger(args)
    if checkpoint is not None:
        logger.best_result = checkpoint['best_result']
        del checkpoint
    print("=> logger created.")
    ###################

    # run
    if args.visualization:

        # build data loader
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0,
                                 pin_memory=True, drop_last=False)

        # build model
        model = get_model(args.model)
        print(f"Model trainable parameters: {count_parameters(model)}")

        # load checkpoint
        if args.ckpt:
            ckpt = torch.load(args.ckpt)
            model.load_state_dict(ckpt['model'])

            trained_epochs = ckpt['epoch']
            global_i = ckpt['global_i']
            best_val = ckpt['best_val']
            train_loss = ckpt['loss']
            try:
                rng = ckpt['rng']
                torch.set_rng_state(rng)
            except KeyError as e:
                print('No rng state in the checkpoint.')
            print(f'Model loaded. '
                  f'Trained epochs: {trained_epochs}; global i: {global_i}; '
                  f'train loss: {train_loss:.04f}; best validation loss: {best_val:.04f}.')
        else:
            print('A model checkpoint must be provided to run the test.')
            exit(0)

        # push model to gpu
        model.to(DEVICE)
        # run task
        run_visualization(model_d, model, test_loader, criteria, args.out_path)
    elif args.iterative:
        # build data loader
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4,
                                 pin_memory=True, drop_last=False)

        # build models
        assert len(args.ckpt_list) != 0, "List of checkpoint files must be provided to run in iterative mode."
        models = []
        offsets = []
        for i, fckpt in enumerate(args.ckpt_list):
            print(f'Loading model {i+1}...')
            model = get_model(args.model)
            ckpt = torch.load(fckpt)

            model.load_state_dict(ckpt['model'])
            model.to(DEVICE)
            models.append(model)
            offsets.append((args.rotation_offsests[i], args.translation_offsests[i]))

            trained_epochs = ckpt['epoch']
            global_i = ckpt['global_i']
            best_val = ckpt['best_val']
            train_loss = ckpt['loss']
            try:
                rng = ckpt['rng']
                torch.set_rng_state(rng)
            except KeyError as e:
                print('No rng state in the checkpoint.')
            print(f'Model loaded. '
                  f'Trained epochs: {trained_epochs}; global i: {global_i}; '
                  f'train loss: {train_loss:.04f}; best validation loss: {best_val:.04f};'
                  f'Offsets: {args.rotation_offsests[i]} degrees, {args.translation_offsests[i]} meters.')

        
        # run task
        run_iterative(model_d, models, test_loader, criteria, offsets, args.out_path)
    
    
    else:
        raise NotImplementedError
