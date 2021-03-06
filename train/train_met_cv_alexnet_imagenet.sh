#!/bin/bash
PROJ_ROOT=$(pwd)

DATA_ROOT=/var/data

CUDA_VISIBLE_DEVICES=${1} \
PYTHONPATH=${PROJ_ROOT}:$PYTHONPATH \
python ${PROJ_ROOT}/train/train.py \
--max_iter 20000 \
--batch_size 1080 \
--snapshot_step 1000 \
--test_step 1000 \
--log_step 1 \
--train_csv_fn ${DATA_ROOT}/MET2/activity_wo_train_k${2}.csv \
--val_csv_fn ${DATA_ROOT}/MET2/activity_wo_val_k${2}.csv \
--test_csv_fn ${DATA_ROOT}/MET2/activity_wo_test.csv \
--img_path_prefix=${DATA_ROOT}/MET2/"" \
--n_joints 14 \
--seed 1701 \
--im_size 227 \
--min_dim 6 \
--shift 0.1 \
--bbox_extension_min 1.2 \
--bbox_extension_max 2.0 \
--coord_normalize \
--fname_index 0 \
--joint_index 1 \
--symmetric_joints "[[8, 9], [7, 10], [6, 11], [2, 3], [1, 4], [0, 5]]" \
--conv_lr 0.0005 \
--fc_lr 0.0005 \
--fix_conv_iter 10000 \
--optimizer adam \
--o_dir ${PROJ_ROOT}/out/met_cv_k${2}_alexnet_imagenet \
--gcn \
--fliplr \
--workers 8 \
--net_type Alexnet \
-s ${PROJ_ROOT}/weights/bvlc_alexnet.tf \
--reset_iter_counter
