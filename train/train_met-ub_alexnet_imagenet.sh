#!/bin/bash
PROJ_ROOT=$(pwd)

DATA_ROOT=/var/data

CUDA_VISIBLE_DEVICES=${1} \
PYTHONPATH=${PROJ_ROOT}:$PYTHONPATH \
python ${PROJ_ROOT}/scripts/train.py \
--max_iter 500000 \
--batch_size 256 \
--snapshot_step 10000 \
--test_step 1000 \
--log_step 10 \
--train_csv_fn ${DATA_ROOT}/MET3/activity-met_n10_ub_new_train.csv \
--val_csv_fn ${DATA_ROOT}/MET3/activity-met_n10_ub_new_train.csv \
--test_csv_fn ${DATA_ROOT}/MET3/activity-met_n10_ub_new_test.csv \
--img_path_prefix=${DATA_ROOT}/MET3 \
--n_joints 8 \
--seed 1701 \
--im_size 227 \
--min_dim 6 \
--shift 0.1 \
--bbox_extension_min 1.2 \
--bbox_extension_max 2.0 \
--coord_normalize \
--fname_index 0 \
--dataset_name met \
--joint_index 13 \
--joint_index_end 29 \
--symmetric_joints "[[2, 3], [1, 4], [0, 5]]" \
--conv_lr 0.0005 \
--fc_lr 0.0005 \
--fix_conv_iter 10000 \
--optimizer adam \
--o_dir ${PROJ_ROOT}/out/met_n10_ub_alexnet_imagenet_gpu${1} \
--gcn \
--fliplr \
--workers 8 \
--net_type Alexnet \
-s ${PROJ_ROOT}/weights/bvlc_alexnet.tf \
--reset_iter_counter
