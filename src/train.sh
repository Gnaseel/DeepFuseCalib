#!/usr/bin/env bash

# use KITTI root directory here as mentioned in the dataset section of README
# https://github.com/simonwu53/NetCalib2-Sensors-Auto-Calibration#2-dataset
KITTI_DIR="/workspace/data/KITTI/"
# KITTI_DIR="/user/SHL/Data_temp/CSW/KITTI"

# activate virtual env if needed
# source /home/username/Projects/venvs/pytorch/bin/activate

# python train.py \
python train_onequeue.py \
--dataset ${KITTI_DIR} \
--batch 2 \
--model 1 \
--epoch 40 \
--lr 1e-5 \
--patience 3 \
--lr_factor 0.5 \
--loss_a 1.25 \
--loss_b 1.75 \
--loss_c 1.0 \
--rotation_offsest 2.0 \
--translation_offsest 0.2 \
# args for resuming (optional)
# --exp_name exp_dir \
# --ckpt /home/username/projects/NetCalib2-Sensors-Auto-Calibration/results/exp_dir/ckpt/Epoch00_val_0.0.tar \
# --ckpt_no_lr