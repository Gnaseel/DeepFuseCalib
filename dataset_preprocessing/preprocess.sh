#!/usr/bin/env bash

BASE_DIR="/workspace/data/KITTI/"
N_WORKER=6
SPLIT="train"

python preprocess.py \
 --data_path ${BASE_DIR} \
 --n_workers ${N_WORKER} \
 --split ${SPLIT} \
 --rotation_offset 0.8 \
 --translation_offset 0.01 \
 --no_cam_depth # use this option to generate LiDAR depth only (no camera depth!)
