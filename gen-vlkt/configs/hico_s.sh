#!/usr/bin/env bash

set -x
EXP_DIR=exps/hico_gen_vlkt_s_r50_dec_3layers_20240422
# python -m torch.distributed.launch \
#         --nproc_per_node=8 \
#         --use_env \
# python  main.py \
#         --pretrained params/detr-r50-pre-2branch-hico.pth \
 #--pretrained /root/autodl-tmp/frp/params/HICO_GEN_VLKT_S.pth \
python  main.py \
        --pretrained /root/autodl-tmp/frp/params/HICO_GEN_VLKT_S.pth \
        --output_dir ${EXP_DIR} \
        --dataset_file hico \
        --hoi_path /root/autodl-tmp/data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --epochs 10 \
        --lr_drop 60 \
        --batch_size 4 \
        --use_nms_filter \
        --ft_clip_with_small_lr \
        --with_clip_label \
        --with_obj_clip_label \
        --with_mimic \
        --mimic_loss_coef 20
