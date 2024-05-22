ulimit -n 4096
set -x
EXP_DIR=exps/hico/hoiclip

#--verb_pth ./tmp/verb.pth \
#            --training_free_enhancement_path ./training_free_ehnahcement/ \
export NCCL_P2P_LEVEL=NVL
export OMP_NUM_THREADS=8
python main.py \
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
            --use_nms_filter \
            --fix_clip \
            --batch_size 4 \
            --pretrained /root/autodl-tmp/frp/params/HICO_GEN_VLKT_S.pth \
            --with_clip_label \
            --with_obj_clip_label \
            --gradient_accumulation_steps 1 \
            --num_workers 8 \
            --opt_sched "multiStep" \
            --dataset_root GEN \
            --model_name HOICLIP \
            --zero_shot_type default \
            --lr 0.00001 \
            --lr_backbone 0.000001 \
            --lr_clip 0.000001 
