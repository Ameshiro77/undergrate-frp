#!/usr/bin/env bash

set -x
export SD_Config="/root/autodl-tmp/DiffHOI/stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
export SD_ckpt="/root/autodl-tmp/DiffHOI/params/v1-5-pruned-emaonly.ckpt"
EXP_DIR=exps/diffhoi_s_hico

#/python -m torch.distributed.launch \
#        --nproc_per_node=1 \
#        --use_env \
python main.py \
        --pretrained params/detr-r50-pre-2branch-hico.pth \
        --output_dir ${EXP_DIR} \
        --model_name diffhoi_s \
        -c configs/DiffHOI_S.py \
        --dataset_file hico \
        --batch_size 2\
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --epochs 10 \
        --lr_drop 60 \
        --use_nms_filter \
        --ft_clip_with_small_lr \
        --with_clip_label \
        --with_obj_clip_label