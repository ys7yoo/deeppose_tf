#!/bin/bash


# input arguments: [GPU_ID] [snapshot filename]

PROJ_ROOT=$(pwd)

DATA_ROOT=/var/data

CUDA_VISIBLE_DEVICES=${1} \
PYTHONPATH=${PROJ_ROOT}:$PYTHONPATH \
python ${PROJ_ROOT}/scripts/train.py \
--max_iter 1000000 \
--batch_size 1024 \
--snapshot_step 10000 \
--test_step 250 \
--log_step 1 \
--train_csv_fn ${DATA_ROOT}/lsp_ext/train_joints.csv \
--test_csv_fn ${DATA_ROOT}/lsp_ext/test_joints.csv \
--val_csv_fn ${DATA_ROOT}/lsp_ext/train_lsp_small_joints.csv \
--img_path_prefix="" \
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
--o_dir ${PROJ_ROOT}/out/lsp_alexnet_imagenet3 \
--gcn \
--fliplr \
--workers 4 \
--net_type Alexnet \
-s ${PROJ_ROOT}/weights/bvlc_alexnet.tf \
--resume  -s ${2} \
--reset_iter_counter


#--resume  -s ${PROJ_ROOT}/out/lsp_alexnet_imagenet/checkpoint-${2}
