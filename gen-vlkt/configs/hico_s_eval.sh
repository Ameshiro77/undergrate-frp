#!/usr/bin/env bash

set -x
EXP_DIR=exps/hico_gen_vlkt_s_r50_dec_3layers_eval
MODEL_PATH=G:/HICO_GEN_VLKT_S.pth
# python -m torch.distributed.launch \
#         --nproc_per_node=1 \
#         --use_env \
#--hoi_path data/hico_20160224_det \
  python      main.py \
        --pretrained ${MODEL_PATH} \
        --output_dir ${EXP_DIR} \
        --dataset_file hico \
        --hoi_path G:\Code_Project\ComputerVision\no_frills_hoi_det-release_v1\HICO\hico_clean\hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --epochs 90 \
        --lr_drop 60 \
        --use_nms_filter \
        --ft_clip_with_small_lr \
        --with_clip_label \
        --with_obj_clip_label \
        --eval \
        --batch_size 1 \
        --device "cuda"