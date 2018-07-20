#!/bin/bash
PROJ_ROOT=$(pwd)
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=${PROJ_ROOT}:$PYTHONPATH \
python ${PROJ_ROOT}/train/train.py \
--max_iter 1000000 \
--batch_size 128 \
--snapshot_step 10000 \
--test_step 250 \
--log_step 1 \
--train_csv_fn ${PROJ_ROOT}/datasets/lsp_ext/train_joints.csv \
--test_csv_fn ${PROJ_ROOT}/datasets/lsp_ext/test_joints.csv \
--val_csv_fn ${PROJ_ROOT}/datasets/lsp_ext/train_lsp_small_joints.csv \
--img_path_prefix="" \
--n_joints 14 \
--seed 1701 \
--im_size 227 \
--min_dim 6 \
--shift 0.1 \
--bbox_extension_min 1.5 \
--bbox_extension_max 2.0 \
--coord_normalize \
--fname_index 0 \
--joint_index 1 \
--symmetric_joints "[[8, 9], [7, 10], [6, 11], [2, 3], [1, 4], [0, 5]]" \
--conv_lr 0.0005 \
--fc_lr 0.0005 \
--fix_conv_iter 0 \
--optimizer adam \
--o_dir ${PROJ_ROOT}/out/lsp_alexnet_scratch \
--gcn \
--fliplr \
--workers 6 \
--net_type Alexnet \
# --resume \
# -s=${PROJ_ROOT}/out/lsp_alexnet_scratch/checkpoint-30000
