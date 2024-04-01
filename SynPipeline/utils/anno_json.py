# 给定提示词(hoi标签)和人物框数组 追加json
import torch, sys
import numpy as np
from matplotlib.patches import Polygon
import json, math

sys.path.append("./")
sys.path.append("../")
# file_name,img_id,annotations["bbox","category_id"],hoi_annotation["subject_id","object_id","category_id","hoi_category_id"]
"""
3.对过滤后的图进行标注。
v_o: 存元组的列表,[(v,o),(v,o)...] 
"""


def get_box_distance(box_1, box_2):  # 获取两个box的中点距离 xyxy格式的
    x1 = (box_1[0] + box_1[2]) / 2
    y1 = (box_1[1] + box_1[3]) / 2
    x2 = (box_2[0] + box_2[2]) / 2
    y2 = (box_2[0] + box_2[2]) / 2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def find_closest_box_id(
    source_box_id: int, tgt: dict, find_sub: True
):  # find_sub决定找人还是找物
    # tgt必须有:boxes box_label_parse_id
    boxes = tgt["boxes"]
    source_box = boxes[source_box_id]
    box_label_parse_id = tgt["box_label_parse_id"]
    dist = [float("inf") for i in range(len(boxes))]
    for idx, target_box in enumerate(boxes):
        if find_sub:
            if idx != source_box_id and box_label_parse_id[idx] == 1:
                bb_dist = get_box_distance(target_box, source_box)
                dist[idx] = bb_dist
        else:
            if idx != source_box_id and box_label_parse_id[idx] != 1:
                bb_dist = get_box_distance(target_box, source_box)
                dist[idx] = bb_dist
    if len(set(dist)) <= 1:  # 如果压根没有人/物 或者就一个框 直接舍弃
        return -1
    return dist.index(min(dist))


def generate_annotation(
    verbs_objs_tuple_list: list, tgt: dict, img_name: str, img_id: int
):
    # =生成标签
    new_anno = {}
    # 名字
    new_anno["file_name"] = img_name
    # img_id
    new_anno["img_id"] = img_id
    # annotations [] 目标的bbox和类别
    o_h, o_w = tgt["original_size"]
    new_anno["annotations"] = []
    for i, bbox in enumerate(tgt["boxes"].to("cpu")):  # to,否则会不在一个设备
        annotation = {}
        unnormbbox = bbox * torch.Tensor([o_w, o_h, o_w, o_h])
        unnormbbox[:2] -= unnormbbox[2:] / 2
        [x, y, w, h] = [int(x) for x in unnormbbox.tolist()]
        xyxy = [x, y, x + w, y + h]
        # print(xyxy)
        annotation["bbox"] = xyxy
        annotation["category_id"] = tgt["box_label_parse_id"][i]
        new_anno["annotations"].append(annotation)

    # hoi_annotation [] , 包含subject id  | object id | category id | hoi category id
    new_anno["hoi_annotation"] = []
    from utils.labels_dict import get_hoi_id

    # 1. 遍历检测到的boxes 取出物框
    is_labeled = [False for i in range(len(tgt["boxes"]))]  # bool数组 记录是否被组合了
    for box_id, box_original_label in enumerate(tgt["box_label_parse_id"]):
        if box_original_label == 1:  # 如果是人框，忽略
            continue
        object_id = box_id  # 说明是物框，换个变量名
        subject_id = find_closest_box_id(object_id, tgt, True)  # 找到离框最近的人
        if subject_id == -1:  # 如果异常检测 即就一个框
            return None
        is_labeled[subject_id] = True
        # 先看这个物体在不在vo列表里，不在的话就分配no_interact
        if box_original_label not in [vo[1] for vo in verbs_objs_tuple_list]:
            hoi_annotation = {}  # 清空
            hoi_annotation["subject_id"] = subject_id
            hoi_annotation["object_id"] = object_id
            hoi_annotation["category_id"] = 58  # 58就是无交互
            hoi_annotation["hoi_category_id"] = get_hoi_id(
                (58, box_original_label)
            )  # 这个是必有得
            if hoi_annotation["hoi_category_id"] == None:
                return None
            new_anno["hoi_annotation"].append(hoi_annotation)
        else:
            for v_o in verbs_objs_tuple_list:  # 找到对应obj_id的动作
                if v_o[1] == box_original_label:
                    hoi_annotation = {}  # 清空
                    hoi_annotation["subject_id"] = subject_id
                    hoi_annotation["object_id"] = object_id
                    hoi_annotation["category_id"] = v_o[0]
                    hoi_annotation["hoi_category_id"] = get_hoi_id(v_o)
                    if hoi_annotation["hoi_category_id"] == None:
                        return None
                    new_anno["hoi_annotation"].append(hoi_annotation)
    # 2. 如果有人框没检测到
    for box_id, box_original_label in enumerate(tgt["box_label_parse_id"]):
        if box_original_label == 1 and is_labeled[box_id] == False:  # 如果没检测到
            subject_id = box_id  # 说明是人框，换个变量名
            object_id = find_closest_box_id(subject_id, tgt, False)
            if object_id == -1:  # 如果异常检测 即就一个框
                return None
            is_labeled[subject_id] = True
            # 先看这个物体在不在vo列表里，不在的话就分配no_interact
            if box_original_label not in [vo[1] for vo in verbs_objs_tuple_list]:
                hoi_annotation = {}  # 清空
                hoi_annotation["subject_id"] = subject_id
                hoi_annotation["object_id"] = object_id
                hoi_annotation["category_id"] = 58  # 58就是无交互
                hoi_annotation["hoi_category_id"] = get_hoi_id(
                    (58, box_original_label)
                )  # 这个是必有得
                if hoi_annotation["hoi_category_id"] == None:
                    return None
                new_anno["hoi_annotation"].append(hoi_annotation)
            else:
                for v_o in verbs_objs_tuple_list:  # 找到对应obj_id的动作
                    if v_o[1] == box_original_label:
                        hoi_annotation = {}  # 清空
                        hoi_annotation["subject_id"] = subject_id
                        hoi_annotation["object_id"] = object_id
                        hoi_annotation["category_id"] = v_o[0]
                        hoi_annotation["hoi_category_id"] = get_hoi_id(v_o)
                        if hoi_annotation["hoi_category_id"] == None:
                            # raise ValueError("不存在对应的hoi的id! "+str(v_o[0])+" "+str(v_o[1]))
                            return None
                        new_anno["hoi_annotation"].append(hoi_annotation)
    print("生成结束")
    # === 生成结束
    if len(new_anno["hoi_annotation"]) == 0:  # 最后一层判断措施
        return None
    return new_anno  # 返回生成的字典


tgt = {
    "boxes": torch.tensor(
        [
            [0.6698, 0.4336, 0.6593, 0.7727],
            [0.6512, 0.7430, 0.3043, 0.2084],
            [0.0801, 0.6507, 0.1595, 0.2835],
            [0.0577, 0.6970, 0.1147, 0.2327],
            [0.3828, 0.8482, 0.5547, 0.1070],
        ],
        device="cuda:0",
    ),
    "size": torch.tensor([512.0, 512.0]),
    "original_size": torch.tensor([512.0, 512.0]),
    "box_label": ["person", "laptop", "cup", "cup", "book"],
    "box_label_parse_id": [1, 73, 47, 47, 84],
}


if __name__ == "__main__":
    new_anno = generate_annotation([(74, 73), (37, 73)], tgt, "1", 1)
    print(new_anno)
