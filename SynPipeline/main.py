import os, sys
import torch, json
import numpy as np
from PIL import Image

sys.path.append("./DINO")
from utils.generate import generate_img
from utils.detect import detect
from utils.anno_json import append_json


class SynPipeline:
    def __init__(self, out_name, config_path, ckpt_path) -> None:
        self.out_name = out_name
        self.config_path = config_path
        self.model_checkpoint_path = ckpt_path

    # 1.生成图片
    def generate(self, prompt, out_name):
        generate_img(prompt, out_name)
        pass

    # 2.检测并过滤
    def detect_and_filter(self, img_name, verbs_objs_tuple_list: list):
        # 检测
        tgt = detect(img_name, self.config_path, self.model_checkpoint_path)
        # 过滤
        #print("tgt:",tgt)
        for v_o in verbs_objs_tuple_list:
            obj_id = v_o[1]
            if obj_id not in tgt["box_label_parse_id"]:
                print(tgt)
                raise ValueError("错误！没检测出生成图片时指定的物体！")
                return None
        return tgt

    # 3.生成标注并追加到已有的标注json里
    def append_anno(self, verbs_objs_tuple_list: list, tgt: dict):
        append_json(verbs_objs_tuple_list, tgt)


if __name__ == "__main__":
    out_name = "/root/autodl-tmp/DiffHOI/SynPipeline/SynDatasets/train_images/Syn_001.jpg"
    model_config_path = "/root/autodl-tmp/DiffHOI/SynPipeline/DINO/config/DINO/DINO_4scale_swin.py"
    model_checkpoint_path = "/root/autodl-tmp/DiffHOI/params/checkpoint0011_4scale_swin.pth"

    from labels_txt.labels import *
    import random
    index = random.randint(1,600)
    v_o = (24,53)
    verb = id_to_verb_dict[v_o[0]]
    obj = id_to_obj_dict[v_o[1]]
    prompt = "a person "+verb+" a "+obj+",high quality,reality,partical view"

    pipeline = SynPipeline(out_name,model_config_path,model_checkpoint_path)
    v_o_list = [v_o]
    print(prompt)
    pipeline.generate(prompt,out_name)
    tgt = pipeline.detect_and_filter("/root/autodl-tmp/DiffHOI/SynPipeline/SynDatasets/train_images/Syn_001.jpg",v_o_list)
    print(tgt)
    pipeline.append_anno(v_o_list,tgt)
