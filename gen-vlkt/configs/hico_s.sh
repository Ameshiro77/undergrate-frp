#!/usr/bin/env bash

set -x
EXP_DIR=exps/20240425_vlkt-90epoch_resume_lr_1e-5
# python -m torch.distributed.launch \
#         --nproc_per_node=8 \
#         --use_env \
# python  main.py \
#         --pretrained params/detr-r50-pre-2branch-hico.pth \
 #--pretrained /root/autodl-tmp/frp/params/HICO_GEN_VLKT_S.pth \
python  main.py \
        --resume /root/autodl-tmp/frp/gen-vlkt/exps/20240425_vlkt-90epoch_resume_lr_1e-5/checkpoint_last.pth \
        --output_dir ${EXP_DIR} \
        --dataset_file hico \
        --hoi_path /root/autodl-tmp/data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --epochs 2 \
        --lr_drop 60 \
        --batch_size 2 \
        --use_nms_filter \
        --ft_clip_with_small_lr \
        --with_clip_label \
        --with_obj_clip_label \
        --with_mimic \
        --mimic_loss_coef 20 \
        --lr 0.00001
