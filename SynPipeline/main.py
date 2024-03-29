import os, sys
import torch, json, random
import numpy as np
from PIL import Image
from labels_txt.labels import id_to_verb_dict,id_to_obj_dict
from labels_txt.hico_text_label import hico_text_label
sys.path.append("./DINO")
from utils.generate import generate_img
from utils.detect import detect
from utils.anno_json import generate_annotation

def random_prompt(v_o:tuple):
    race = random.choice(["asian", "black", "hispanic"])
    human = random.choice(["boy", "teenager", "man", "young man", "woman", "young woman"])
    quality = random.choice(["best quality","masterpiece"])
    details = random.choice(["Professional","Vivid Colors"])
    scene = random.choice(["spacious", "urban", "rustic"])
    shooting = random.choice(["DSLR","grainy","4K"])
    shooting2 = random.choice(["warm lighting","blue hour","backlit"])
    shooting3 = random.choice(["partial view","back view"])
    shooting4 = random.choice(["Canon Eos5D","iphone 12"])
    # prompt_prefix = "Photo of a " + race + " " + human + " " + verb + " a " + obj + ","
    prompt_prefix = hico_text_label[v_o].replace("person",race+" "+human)+","
    prompt_suffix = quality + "," + details + "," + scene + "," + \
    shooting + "," + shooting2 + "," + shooting3 + "," + shooting4
    return prompt_prefix + prompt_suffix
    
class SynPipeline:
    def __init__(self, config_path, ckpt_path) -> None:
        self.config_path = config_path
        self.model_checkpoint_path = ckpt_path

    # 1.生成图片
    def generate(self, prompt):
         # == 随机prompt
        #index = random.randint(1, 600)
        #verb = id_to_verb_dict[v_o[0]]
        #obj = id_to_obj_dict[v_o[1]]
        return generate_img(prompt) #返回PIL image数组

    # 2.检测并过滤，然后  3.标注
    def detect_and_filter_and_anno(self, imgs, verbs_objs_tuple_list: list,out_dir):
        # == 读取json 为了之后的标注
        with open("./SynDatasets/annotations/train_val.json","r") as f:
            anno = json.load(f)
        # file_name,id
        files_num = len(anno)
        formatted_name = "Syn_train_" + '{:06d}'.format(files_num + 1) + ".jpg"
        # == 对每个图像检测
        for img in imgs:
            tgt = detect(img, self.config_path, self.model_checkpoint_path)
            # 过滤
            # print("tgt:",tgt)
            # == 检查是否检测出了指定的物体
            for v_o in verbs_objs_tuple_list:
                obj_id = v_o[1]
                if obj_id not in tgt["box_label_parse_id"]:
                    #print(tgt)
                    #raise ValueError("错误！没检测出生成图片时指定的物体！")
                    break
            # == 如果检测出来就做标注,并保存
            new_anno = generate_annotation(verbs_objs_tuple_list,tgt,formatted_name,files_num + 1) #先获得标注框
            if new_anno != None:
                img.save(formatted_name)
                anno.append(new_anno)
                json.dump(anno, f, indent=4)
        f.close()

if __name__ == "__main__":
    out_dir = (
        "./SynDatasets/train_images"
    )
    model_config_path = (
        "/root/autodl-tmp/DiffHOI/SynPipeline/DINO/config/DINO/DINO_4scale_swin.py"
    )
    model_checkpoint_path = (
        "/root/autodl-tmp/DiffHOI/params/checkpoint0011_4scale_swin.pth"
    )

    # ==== pipeline
    v_o = random.choice(list(hico_text_label.keys()))
    prompt = random_prompt(v_o)
    pipeline = SynPipeline(model_config_path, model_checkpoint_path)
    v_o_list = [v_o]
    print(prompt)
    imgs = pipeline.generate(prompt)
    pipeline.detect_and_filter_and_anno(imgs,v_o_list,out_dir)
