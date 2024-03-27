import os, sys
import torch, json
import numpy as np
from PIL import Image

sys.path.append("./DINO")
#from generate import generate_img
from utils.detect import detect
from utils.anno_json import append_json


class SynPipeline:
    def __init__(self, out_name, config_path, ckpt_path) -> None:
        self.out_name = out_name
        self.config_path = config_path
        self.model_checkpoint_path = ckpt_path

    # 1.生成图片
    def generate(self, prompt, out_name):
    #    generate_img(prompt, out_name + "jpg")
        pass

    # 2.检测并过滤
    def detect_and_filter(self, img_name, verbs_objs_tuple_list: list):
        # 检测
        tgt = detect(img_name, self.config_path, self.model_checkpoint_path)
        # 过滤
        for v_o in verbs_objs_tuple_list:
            obj_id = v_o[1]
            if obj_id not in tgt["box_label_parse_id"]:
                print("错误！没检测出生成图片时指定的物体！")
                return None
        return tgt

    # 3.生成标注并追加到已有的标注json里
    def append_anno(self, verbs_objs_tuple_list: list, tgt: dict):
        append_json(verbs_objs_tuple_list, tgt)


if __name__ == "__main__":
    out_name = "./output10"
    model_config_path = (
        r"/root/autodl-tmp/DiffHOI/SynPipeline/DINO/config/DINO/DINO_4scale_swin.py"
    )
    model_checkpoint_path = (
        r"/root/autodl-tmp/DiffHOI/params/checkpoint0011_4scale_swin.pth"
    )

    pipeline = SynPipeline()
    v_o = [(5, 1), (10, 1)]
    tgt = pipeline.detect_and_filter("output.jpg",v_o)
    pipeline.append_anno(v_o,tgt)
