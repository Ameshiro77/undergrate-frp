import argparse
import datetime
from datasets.hico_text_label import hico_text_label
from util.topk import top_k
import json
import random
import time,cv2
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset
from engine import train_one_epoch, evaluate_hoi
from models import build_model
import os
from collections import defaultdict
from util.misc import nested_tensor_from_tensor_list


class HICOEvaluator:
    def predict(self, preds, args):
        self.overlap_iou = 0.5
        self.max_hois = 100

        self.use_nms_filter = args.use_nms_filter
        self.thres_nms = args.thres_nms
        self.nms_alpha = args.nms_alpha
        self.nms_beta = args.nms_beta

        self.use_score_thres = False
        self.thres_score = 1e-5

        self.use_soft_nms = False
        self.soft_nms_sigma = 0.5
        self.soft_nms_thres_score = 1e-11

        self.fp = defaultdict(list)
        self.tp = defaultdict(list)
        self.score = defaultdict(list)
        self.sum_gts = defaultdict(lambda: 0)
        self.gt_triplets = []

        self.preds = []
        self.hico_triplet_labels = list(hico_text_label.keys())
        self.hoi_obj_list = []
        for hoi_pair in self.hico_triplet_labels:
            self.hoi_obj_list.append(hoi_pair[1])

        for index, img_preds in enumerate(preds):
            img_preds = {k: v.to("cpu").numpy() for k, v in img_preds.items()}
            bboxes = [{"bbox": list(bbox)} for bbox in img_preds["boxes"]]
            obj_scores = img_preds["obj_scores"] * img_preds["obj_scores"]
            hoi_scores = (
                img_preds["hoi_scores"] + obj_scores[:, self.hoi_obj_list]
            )  # 64*600

            hoi_labels = np.tile(
                np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1)
            )  # 64*600,[0~599 0~599 ..]
            subject_ids = np.tile(img_preds["sub_ids"], (hoi_scores.shape[1], 1)).T
            object_ids = np.tile(img_preds["obj_ids"], (hoi_scores.shape[1], 1)).T

            hoi_scores = hoi_scores.ravel()  # ravel 拉成一维数组
            hoi_labels = hoi_labels.ravel()
            subject_ids = subject_ids.ravel()
            object_ids = object_ids.ravel()

            topk_hoi_scores = top_k(list(hoi_scores), self.max_hois)  # 用了堆排序
            topk_indexes = np.array(
                [np.where(hoi_scores == score)[0][0] for score in topk_hoi_scores]
            )

            if len(subject_ids) > 0:
                hois = [
                    {
                        "subject_id": subject_id,
                        "object_id": object_id,
                        "category_id": category_id,
                        "score": score,
                    }
                    for subject_id, object_id, category_id, score in zip(
                        subject_ids[topk_indexes],
                        object_ids[topk_indexes],
                        hoi_labels[topk_indexes],
                        topk_hoi_scores,
                    )
                ]
                hois = hois[: self.max_hois]
            else:
                hois = []

            self.preds.append(
                {"filename": "filename", "predictions": bboxes, "hoi_prediction": hois}
            )

        if self.use_nms_filter:
            print("eval use_nms_filter ...")
            self.preds = self.triplet_nms_filter(self.preds)
        return self.preds

    def triplet_nms_filter(self, preds):
        preds_filtered = []
        for img_preds in preds:
            pred_bboxes = img_preds["predictions"]
            pred_hois = img_preds["hoi_prediction"]
            all_triplets = {}
            for index, pred_hoi in enumerate(pred_hois):
                triplet = pred_hoi["category_id"]

                if triplet not in all_triplets:
                    all_triplets[triplet] = {
                        "subs": [],
                        "objs": [],
                        "scores": [],
                        "indexes": [],
                    }
                all_triplets[triplet]["subs"].append(
                    pred_bboxes[pred_hoi["subject_id"]]["bbox"]
                )
                all_triplets[triplet]["objs"].append(
                    pred_bboxes[pred_hoi["object_id"]]["bbox"]
                )
                all_triplets[triplet]["scores"].append(pred_hoi["score"])
                all_triplets[triplet]["indexes"].append(index)

            all_keep_inds = []
            for triplet, values in all_triplets.items():
                subs, objs, scores = values["subs"], values["objs"], values["scores"]
                if self.use_soft_nms:
                    keep_inds = self.pairwise_soft_nms(
                        np.array(subs), np.array(objs), np.array(scores)
                    )
                else:
                    keep_inds = self.pairwise_nms(
                        np.array(subs), np.array(objs), np.array(scores)
                    )

                if self.use_score_thres:
                    sorted_scores = np.array(scores)[keep_inds]
                    keep_inds = np.array(keep_inds)[sorted_scores > self.thres_score]

                keep_inds = list(np.array(values["indexes"])[keep_inds])
                all_keep_inds.extend(keep_inds)

            preds_filtered.append(
                {
                    "filename": img_preds["filename"],
                    "predictions": pred_bboxes,
                    "hoi_prediction": list(
                        np.array(img_preds["hoi_prediction"])[all_keep_inds]
                    ),
                }
            )

        return preds_filtered
    
    def pairwise_nms(self, subs, objs, scores):
        sx1, sy1, sx2, sy2 = subs[:, 0], subs[:, 1], subs[:, 2], subs[:, 3]
        ox1, oy1, ox2, oy2 = objs[:, 0], objs[:, 1], objs[:, 2], objs[:, 3]

        sub_areas = (sx2 - sx1 + 1) * (sy2 - sy1 + 1)
        obj_areas = (ox2 - ox1 + 1) * (oy2 - oy1 + 1)

        order = scores.argsort()[::-1]

        keep_inds = []
        while order.size > 0:
            i = order[0]
            keep_inds.append(i)

            sxx1 = np.maximum(sx1[i], sx1[order[1:]])
            syy1 = np.maximum(sy1[i], sy1[order[1:]])
            sxx2 = np.minimum(sx2[i], sx2[order[1:]])
            syy2 = np.minimum(sy2[i], sy2[order[1:]])

            sw = np.maximum(0.0, sxx2 - sxx1 + 1)
            sh = np.maximum(0.0, syy2 - syy1 + 1)
            sub_inter = sw * sh
            sub_union = sub_areas[i] + sub_areas[order[1:]] - sub_inter

            oxx1 = np.maximum(ox1[i], ox1[order[1:]])
            oyy1 = np.maximum(oy1[i], oy1[order[1:]])
            oxx2 = np.minimum(ox2[i], ox2[order[1:]])
            oyy2 = np.minimum(oy2[i], oy2[order[1:]])

            ow = np.maximum(0.0, oxx2 - oxx1 + 1)
            oh = np.maximum(0.0, oyy2 - oyy1 + 1)
            obj_inter = ow * oh
            obj_union = obj_areas[i] + obj_areas[order[1:]] - obj_inter

            ovr = np.power(sub_inter / sub_union, self.nms_alpha) * np.power(obj_inter / obj_union, self.nms_beta)
            inds = np.where(ovr <= self.thres_nms)[0]

            order = order[inds + 1]
        return keep_inds



def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--lr_clip", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--lr_drop", default=100, type=int)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )
    # * Backbone
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=3,
        type=int,
        help="Number of stage1 decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--num_queries", default=64, type=int, help="Number of query slots"
    )
    parser.add_argument("--pre_norm", action="store_true")

    # * Segmentation
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Train segmentation head if the flag is provided",
    )

    # HOI
    parser.add_argument(
        "--hoi", action="store_true", help="Train for HOI if the flag is provided"
    )
    parser.add_argument(
        "--num_obj_classes", type=int, default=80, help="Number of object classes"
    )
    parser.add_argument(
        "--num_verb_classes", type=int, default=117, help="Number of verb classes"
    )
    parser.add_argument(
        "--pretrained", type=str, default="", help="Pretrained model path"
    )
    parser.add_argument("--subject_category_id", default=0, type=int)
    parser.add_argument(
        "--verb_loss_type",
        type=str,
        default="focal",
        help="Loss type for the verb classification",
    )

    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )
    parser.add_argument(
        "--with_mimic", default=True, action="store_true", help="Use clip feature mimic"
    )
    # * Matcher
    parser.add_argument(
        "--set_cost_class",
        default=1,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=2.5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=1,
        type=float,
        help="giou box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_obj_class",
        default=1,
        type=float,
        help="Object class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_verb_class",
        default=1,
        type=float,
        help="Verb class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_hoi", default=1, type=float, help="Hoi class coefficient"
    )

    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=2.5, type=float)
    parser.add_argument("--giou_loss_coef", default=1, type=float)
    parser.add_argument("--obj_loss_coef", default=1, type=float)
    parser.add_argument("--verb_loss_coef", default=2, type=float)
    parser.add_argument("--hoi_loss_coef", default=2, type=float)
    parser.add_argument("--mimic_loss_coef", default=20, type=float)
    parser.add_argument("--alpha", default=0.5, type=float, help="focal loss alpha")
    parser.add_argument(
        "--eos_coef",
        default=0.1,
        type=float,
        help="Relative classification weight of the no-object class",
    )

    # dataset parameters
    parser.add_argument("--dataset_file", default="hico")
    parser.add_argument("--coco_path", type=str)
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--hoi_path", type=str)

    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=2, type=int)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    # hoi eval parameters
    parser.add_argument(
        "--use_nms_filter",
        action="store_true",
        default=True,
        help="Use pair nms filter, default not use",
    )
    parser.add_argument("--thres_nms", default=0.7, type=float)
    parser.add_argument("--nms_alpha", default=1, type=float)
    parser.add_argument("--nms_beta", default=0.5, type=float)
    parser.add_argument("--json_file", default="results.json", type=str)

    # clip
    parser.add_argument(
        "--ft_clip_with_small_lr",
        action="store_true",
        default= True,
        help="Use smaller learning rate to finetune clip weights",
    )
    parser.add_argument(
        "--with_clip_label", action="store_true", default= True,help="Use clip to classify HOI"
    )
    parser.add_argument(
        "--early_stop_mimic", action="store_true", help="stop mimic after step"
    )
    parser.add_argument(
        "--with_obj_clip_label", action="store_true", default= True,help="Use clip to classify object"
    )
    parser.add_argument(
        "--clip_model", default="ViT-B/32", help="clip pretrained model path"
    )
    parser.add_argument("--fix_clip", action="store_true", help="")
    parser.add_argument("--clip_embed_dim", default=512, type=int)

    # zero shot type
    parser.add_argument(
        "--zero_shot_type",
        default="default",
        help="default, rare_first, non_rare_first, unseen_object, unseen_verb",
    )
    parser.add_argument("--del_unseen", action="store_true", help="")

    # DATASET
    parser.add_argument("--dataset_json", default="hico", type=str, help="if use syn")
    return parser

def get_prediction(img):
    parser = argparse.ArgumentParser("visualization", parents=[get_args_parser()])
    args = parser.parse_args()
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    print("****************")
    print(model)
    print("****************")

    import cv2

    if 1:
        print("start")
        args.hoi_path = r"G:\Code_Project\ComputerVision\no_frills_hoi_det-release_v1\HICO\hico_clean\hico_20160224_det"
        # dataset_train = build_dataset(image_set='train', args=args)   #继承dataset类。args指明HICO/VCOCO
        # dataset_val = build_dataset(image_set='val', args=args)
        # sampler_train = torch.utils.data.RandomSampler(dataset_train)
        # sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        # batch_sampler_train = torch.utils.data.BatchSampler(
        #     sampler_train, args.batch_size, drop_last=True)
        # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
        #                             collate_fn=utils.collate_fn, num_workers=args.num_workers)
        # data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
        #                          drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        checkpoint = torch.load("G:\HICO_GEN_VLKT_S.pth", map_location="cpu")
        model_without_ddp = model
        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
        with torch.no_grad():
            import torchvision.transforms as T
            h, w = img.shape[0], img.shape[1]
            tf = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            img = tf(img)
            src = nested_tensor_from_tensor_list([img]).to("cuda")
            out = model(src)
            orig_target_sizes = torch.unsqueeze(
                torch.as_tensor([int(h), int(w)]), dim=0
            )
            results = postprocessors["hoi"](out, orig_target_sizes)
            preds = []
            #print(results)
            preds.extend(results)
            evaluator = HICOEvaluator()
            predict = evaluator.predict(preds, args)
            # predict:[
            # filename
            # predictions: [{'bbox':[1 1 1 1]},...]
            # hoi_prediction: [{'subject_id'/object_id/category_id/score}]
            #]
            return predict
        
def draw(img, tgt_dict):
    so_count = {}
    blk = np.zeros(img.shape, np.uint8)  # 透明背景用的
    # 画box
    boxes = tgt_dict["annotations"]  # 后面还有用
    for box_category in boxes:
        box = box_category["bbox"]
        category_id = box_category["category_id"]
        xy1 = (box[0], box[1])
        xy2 = (box[2], box[3])
        color = (np.random.random(3) * 0.6 + 0.4) * 255
        # print(color)
        cv2.rectangle(img, xy1, xy2, color, 2)  # 画人物的框
        text = id_to_obj_dict[category_id]
        cv2.rectangle(  # 用于text的背景框
            blk,
            (xy1[0], xy1[1]),
            (xy1[0] + len(text) * 10, xy1[1] + 25),
            color,
            -1,
        )
        cv2.putText(
            blk,
            text,
            (xy1[0], xy1[1] + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0, 0),
            2,
        )

    # 画交互
    for hoi in tgt_dict["hoi_annotation"]:
        xyxy1 = boxes[hoi["subject_id"]]["bbox"]  # 得到四个点
        xyxy2 = boxes[hoi["object_id"]]["bbox"]
        xy1 = (
            int(sum(xyxy1[0::2]) / 2),
            int(sum(xyxy1[1::2]) / 2),
        )  # 找到两个box的中点坐标
        xy2 = (int(sum(xyxy2[0::2]) / 2), int(sum(xyxy2[1::2]) / 2))
        # 如果是no interaction ， 画一条红线 ,无文字
        if hoi["category_id"] == 58:
            cv2.line(blk, xy1, xy2, (0, 0, 255), 3)
        else:
            s_o = (
                hoi["subject_id"],
                hoi["object_id"],
            )  # 记录主体客体对出现次数 调整text用的
            if so_count.get(s_o) == None:
                so_count[s_o] = 0
            else:
                so_count[s_o] = so_count[s_o] + 1
            text_pos = (
                int((xy1[0] + xy2[0]) / 2),
                int((xy1[1] + xy2[1]) / 2) + 28 * so_count[s_o],
            )  # 放文字的基准点
            text = id_to_verb_dict[hoi["category_id"]]
            cv2.rectangle(  # 用于text的背景框
                blk,
                (text_pos[0], text_pos[1]),
                (text_pos[0] + len(text) * 10, text_pos[1] + 25),
                (255, 255, 255),
                -1,
            )
            cv2.putText(
                blk,
                text,
                (text_pos[0], text_pos[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0, 0),
                2,
            )
            cv2.line(blk, xy1, xy2, (0, 255, 0), 3)
        # 画交互点
        cv2.circle(
            blk,
            (int((xy1[0] + xy2[0]) / 2), int((xy1[1] + xy2[1]) / 2)),
            3,
            (255, 0, 0),
            -1,
        )
        cv2.circle(
            blk,
            (xy1[0], xy1[1]),
            3,
            (255, 0, 0),
            -1,
        )
        cv2.circle(
            blk,
            (xy2[0], xy2[1]),
            3,
            (255, 0, 0),
            -1,
        )
    img = cv2.addWeighted(blk, 2, img, 0.85, 0)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser("visualization", parents=[get_args_parser()])
    img = cv2.imread("./demo.jpg")
    prediction = get_prediction(img)[0]
    hoi_predictions = []
    for i in range(3):
        hoi_prediction = {}
        sub_id = prediction['hoi_prediction'][i]["subject_id"]
        obj_id = prediction['hoi_prediction'][i]["object_id"]
        hoi_id = prediction['hoi_prediction'][i]["category_id"]
        hoi_prediction['sub_bbox'] = prediction['predictions'][sub_id]["bbox"]
        hoi_prediction['obj_bbox'] = prediction['predictions'][obj_id]["bbox"]
        hoi_prediction['hoi_id'] = prediction['hoi_prediction'][i]["category_id"]
        hoi_predictions.append(hoi_prediction)
    print(hoi_predictions)
        
    
