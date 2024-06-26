#!/usr/bin/env bash

set -x
export SD_Config="/root/autodl-tmp/stable-diffusion-main/configs/stable-diffusion/v1-inference.yaml"
export SD_ckpt="/root/autodl-tmp/DiffHOI/params/v1-5-pruned-emaonly.ckpt"
EXP_DIR=exps/diffhoi_s_vcoco

# python -m torch.distributed.launch \
#         --nproc_per_node=8 \
#         --use_env \
 python       main.py \
        --pretrained params/detr-r50-pre-2branch-vcoco.pth \
        --model_name diffhoi_s \
        -c configs/DiffHOI_S.py \
        --output_dir ${EXP_DIR} \
        --dataset_file vcoco \
        --hoi_path data/v-coco \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --batch_size 2 \
        --epochs 50 \
        --lr_drop 60 \
        --use_nms_filter \
        --ft_clip_with_small_lr \
        --with_clip_label \
        --with_obj_clip_label
