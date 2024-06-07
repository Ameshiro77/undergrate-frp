import argparse
import datetime
from datasets.hico_text_label import hico_text_label
from util.topk import top_k
import json
import random
import time, cv2
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from labels_txt.labels import id_to_obj_dict, id_to_verb_dict, id_to_hoi_dict
import datasets
import util.misc as utils
from datasets import build_dataset
from engine import train_one_epoch, evaluate_hoi
from models import build_model
import os
from collections import defaultdict
from util.misc import nested_tensor_from_tensor_list


class HICOEvaluator:
    def __init__(self, args):
        self.overlap_iou = 0.5
        self.max_hois = 100

        self.use_nms_filter = args.use_nms_filter
        self.thres_nms = 0.6
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

    def predict(self, preds):
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

            ovr = np.power(sub_inter / sub_union, self.nms_alpha) * np.power(
                obj_inter / obj_union, self.nms_beta
            )
            inds = np.where(ovr <= self.thres_nms)[0]

            order = order[inds + 1]
        return keep_inds


def get_prediction(img):
    from args import get_args_parser

    parser = argparse.ArgumentParser("visualization", parents=[get_args_parser()])
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.backends.cudnn.deterministic = True
    print("开始构造模型")
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    print("构造完毕")
    args.hoi_path = r"G:\Code_Project\ComputerVision\no_frills_hoi_det-release_v1\HICO\hico_clean\hico_20160224_det"
    #checkpoint = torch.load("G:\HICO_GEN_VLKT_S.pth", map_location="cpu")
    checkpoint = torch.load("G:\checkpoint_best.pth", map_location="cpu")
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
        orig_target_sizes = torch.unsqueeze(torch.as_tensor([int(h), int(w)]), dim=0)
        results = postprocessors["hoi"](out, orig_target_sizes)
        preds = []
        # print(results)
        preds.extend(results)
        evaluator = HICOEvaluator(args)
        predict = evaluator.predict(preds)
        # predict:[
        # filename
        # predictions: [{'bbox':[1 1 1 1]},...]
        # hoi_prediction: [{'subject_id'/object_id/category_id/score}]
        # ]
    return predict


def draw_and_save(img, preds, full_path, nohoi_path):
    blk = np.zeros(img.shape, np.uint8)  # 透明背景用的
    new_preds = []
    # 画box
    for index, pred in enumerate(preds):
        # 画bbox
        for i in range(2):
            if i == 0:
                box = pred["sub_bbox"]
                obj_id = 1
            else:
                box = pred["obj_bbox"]
                obj_id = id_to_hoi_dict[pred["hoi_id"]][1]
            xy1 = (int(box[0]), int(box[1]))
            xy2 = (int(box[2]), int(box[3]))
            # print(color)
            color = (np.random.random(3) * 0.6 + 0.4) * 255
            cv2.rectangle(img, xy1, xy2, color, 2)  # 画人物的框
            text = id_to_obj_dict[obj_id]
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
            
        # 转化为前端要的格式
        new_pred = {}
        new_pred["sub_bbox"] = (
            "[" + ",".join([str(int(x)) for x in pred["sub_bbox"]]) + "]"
        )
        new_pred["obj_bbox"] = (
            "[" + ",".join([str(int(x)) for x in pred["obj_bbox"]]) + "]"
        )
        obj_id, verb_id = (
            id_to_hoi_dict[pred["hoi_id"]][1],
            id_to_hoi_dict[pred["hoi_id"]][0],
        )
        obj_name, verb_name = id_to_obj_dict[obj_id], id_to_verb_dict[verb_id]
        new_pred["hoi_name"] = verb_name + "-" + obj_name
        new_pred["score"] = str(pred["score"])
        new_preds.append(new_pred)
        # ==
        
    cv2.imwrite(nohoi_path, img)
    print("new_preds", new_preds)
    # 画hoi
    for index, pred in enumerate(preds):
        hoi_id = pred["hoi_id"]
        verb_id = id_to_hoi_dict[hoi_id][0]
        xyxy1 = pred["sub_bbox"]  # 得到四个点
        xyxy2 = pred["obj_bbox"]
        xy1 = (
            int(sum(xyxy1[0::2]) / 2),
            int(sum(xyxy1[1::2]) / 2),
        )  # 找到两个box的中点坐标
        xy2 = (int(sum(xyxy2[0::2]) / 2), int(sum(xyxy2[1::2]) / 2))
        # 如果是no interaction ， 画一条红线 ,无文字
        if verb_id == 58:
            cv2.line(blk, xy1, xy2, (0, 0, 255), 3)
        else:
            text_pos = (
                int((xy1[0] + xy2[0]) / 2),
                int((xy1[1] + xy2[1]) / 2) + index * 20,
            )  # 放文字的基准点
            text = id_to_verb_dict[verb_id]
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
    cv2.imwrite(full_path, img)

    return img, new_preds


def predict(img, rank, threshold, full_path, nohoi_path):
    prediction = get_prediction(img)[0]
    # print(prediction)
    curr_id = -1
    # pred_for_top_hoi = [
    #     {
    #         "sub_bbox": [209.5744, 32.900505, 444.9329, 294.5358],
    #         "obj_bbox": [57.74666, 99.23865, 574.8442, 404.72195],
    #         "hoi_id": 153,
    #         "score": 1.7029605,
    #     },
    #     {
    #         "sub_bbox": [209.5744, 32.900505, 444.9329, 294.5358],
    #         "obj_bbox": [57.74666, 99.23865, 574.8442, 404.72195],
    #         "hoi_id": 154,
    #         "score": 1.6451001,
    #     },
    #     {
    #         "sub_bbox": [209.5744, 32.900505, 444.9329, 294.5358],
    #         "obj_bbox": [57.74666, 99.23865, 574.8442, 404.72195],
    #         "hoi_id": 152,
    #         "score": 1.5870278,
    #     },
    # ]
    pred_for_top_hoi = []
    for i, pred in enumerate(prediction["hoi_prediction"]):
        hoi_id = pred["category_id"]
        if curr_id == hoi_id:
            continue
        curr_id = hoi_id
        # 格式
        hoi_prediction = {}
        sub_id = prediction["hoi_prediction"][i]["subject_id"]
        obj_id = prediction["hoi_prediction"][i]["object_id"]
        hoi_id = prediction["hoi_prediction"][i]["category_id"]
        hoi_prediction["sub_bbox"] = prediction["predictions"][sub_id]["bbox"]
        hoi_prediction["obj_bbox"] = prediction["predictions"][obj_id]["bbox"]
        hoi_prediction["hoi_id"] = prediction["hoi_prediction"][i]["category_id"]
        hoi_prediction["score"] = prediction["hoi_prediction"][i]["score"]
        if hoi_prediction["score"] < threshold:
            continue
        pred_for_top_hoi.append(hoi_prediction)

    for i, pred in enumerate(pred_for_top_hoi[:rank]):
        print(i, pred, " ")

    drown_img, info = draw_and_save(img, pred_for_top_hoi[:rank], full_path, nohoi_path)
    return drown_img, info


if __name__ == "__main__":
    img = cv2.imread("./demo_2.jpg")
    predict(img, 1, 1)
