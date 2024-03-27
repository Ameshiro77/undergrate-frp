# 给定提示词(hoi标签)和人物框数组 追加json
import torch,sys
import numpy as np
from matplotlib.patches import Polygon
import json,math
sys.path.append("./")
#file_name,img_id,annotations["bbox","category_id"],hoi_annotation["subject_id","object_id","category_id","hoi_category_id"]
"""
3.对过滤后的图进行标注。
v_o: 存元组的列表,[(v,o),(v,o)...] 
"""
def get_box_distance(box_1,box_2): #获取两个box的中点距离 xyxy格式的
    x1 = (box_1[0]+box_1[2])/2
    y1 = (box_1[1]+box_1[3])/2
    x2 = (box_2[0]+box_2[2])/2
    y2 = (box_2[0]+box_2[2])/2
    return math.sqrt( (x1-x2)**2 + (y1-y2)**2 )

def find_closest_box_id(source_box_id:int,tgt:dict,find_sub:True): #find_sub决定找人还是找物
    # tgt必须有:boxes box_label_parse_id
    boxes = tgt["boxes"]
    source_box = boxes[source_box_id]
    box_label_parse_id = tgt["box_label_parse_id"]
    dist = [float('inf') for i in range(len(boxes))]
    for idx,target_box in enumerate(boxes):
        if find_sub:
            if idx!=source_box_id and box_label_parse_id[idx] == 1:
                bb_dist = get_box_distance(target_box,source_box)
                dist[idx] = bb_dist
        else:
            if idx!=source_box_id and box_label_parse_id[idx] != 1:
                bb_dist = get_box_distance(target_box,source_box)
                dist[idx] = bb_dist
    return dist.index(min(dist))


def append_json(verbs_objs_tuple_list:list, tgt: dict): 
    # == 读取json
    with open("./SynDatasets/annotations/train_val.json","r") as f:
        anno = json.load(f)
    
    # == 此段生成标签
    new_anno = {}
    # file_name
    files_num = len(anno)
    formatted_number = '{:06d}'.format(files_num + 1)
    new_anno["file_name"] = "Syn_train_" + formatted_number + ".jpg"

    # img_id
    new_anno["img_id"] = files_num + 1

    # annotations [] 目标的bbox和类别
    H, W = tgt['size']
    new_anno["annotations"] = []
    for i ,bbox in enumerate(tgt["boxes"].to("cpu")):  #to,否则会不在一个设备
        annotation = {}
        unnormbbox = bbox * torch.Tensor([W, H, W, H])
        unnormbbox[:2] -= unnormbbox[2:] / 2
        [x, y, w, h] = [int(x) for x in unnormbbox.tolist()]
        xyxy = [x , y , x + w , y + h]
        # print(xyxy)
        annotation["bbox"] = xyxy
        annotation["category_id"] = tgt['box_label_parse_id'][i]
        new_anno["annotations"].append(annotation)

    # hoi_annotation [] , 包含subject id  | object id | category id | hoi category id  
    new_anno["hoi_annotation"] = []
    from utils.labels_dict import get_hoi_id
    # 1. 遍历检测到的boxes 取出物框
    is_labeled = [False for i in range(len(tgt["boxes"]))]  #bool数组 记录是否被组合了
    for box_id,box_original_label in enumerate(tgt['box_label_parse_id']):
        if box_original_label == 1: #如果是人框，忽略
            continue 
        object_id = box_id #说明是物框，换个变量名
        subject_id = find_closest_box_id(object_id,tgt,True) #找到离框最近的人
        is_labeled[subject_id] = True
        for v_o in verbs_objs_tuple_list: #找到对应obj_id的动作
            if v_o[1] == box_original_label:
                hoi_annotation = {}  #清空
                hoi_annotation["subject_id"] = subject_id
                hoi_annotation["object_id"] = object_id
                hoi_annotation["category_id"] = v_o[0]
                hoi_annotation["hoi_category_id"] = get_hoi_id(v_o)
                if hoi_annotation["hoi_category_id"] == None:
                    raise ValueError("不存在对应的hoi的id! "+str(v_o[0])+" "+str(v_o[1]))
                new_anno["hoi_annotation"].append(hoi_annotation)
    # 2. 如果有人框没检测到
    for box_id,box_original_label in enumerate(tgt['box_label_parse_id']):
        if box_original_label == 1 and is_labeled[box_id] == False: 
            object_id = find_closest_box_id(subject_id,tgt,False)
            is_labeled[subject_id] = True
            for v_o in verbs_objs_tuple_list: #找到对应obj_id的动作
                if v_o[1] == box_original_label:
                    hoi_annotation = {}  #清空
                    hoi_annotation["subject_id"] = subject_id
                    hoi_annotation["object_id"] = object_id
                    hoi_annotation["category_id"] = v_o[0]
                    hoi_annotation["hoi_category_id"] = get_hoi_id(v_o)
                    if hoi_annotation["hoi_category_id"] == None:
                        raise ValueError("不存在对应的hoi的id! "+str(v_o[0])+" "+str(v_o[1]))
                    new_anno["hoi_annotation"].append(hoi_annotation)
    print("生成结束")
    # === 生成结束
    
    # == 追加并写回
    anno.append(new_anno)
    with open('./SynDatasets/annotations/train_val.json', 'w') as f:
        json.dump(anno, f, indent=4)
    f.close()



tgt = {
    "boxes": torch.tensor(
        [[0.6207, 0.5451, 0.4399, 0.7578], [0.3205, 0.6232, 0.6407, 0.5138]]),
    "size": torch.tensor([800.0, 800.0]),
    "box_label": ["person", "car"],
    "box_label_parse_id": [1, 3]
}


if __name__ == "__main__": 
     print(1)
     append_json([(5,1)],tgt)
