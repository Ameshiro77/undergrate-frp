import os, sys
from diffusers import DiffusionPipeline, StableDiffusionPipeline
import torch, json, random
import numpy as np
from PIL import Image
from labels_txt.labels import id_to_verb_dict,id_to_obj_dict,valid_obj_ids,id_to_hoi_dict,original_labels_dict,hoi_to_id_dict
from labels_txt.hico_text_label import hico_text_label,hico_unseen_index,hico_obj_text_label
from labels_txt.vo_pairs import vo_pairs,multi_hoi
sys.path.append("./DINO")
from utils.detect import detect
from utils.anno_json import generate_annotation

def get_verb(v_o): #接受一个元组
    prompt = hico_text_label.get(v_o).split()[5:]
    str = ""
    for word in prompt:
        if word == "a" or word == "an":
            break
        str = str + word + " "
    return str[:-1]

def get_noun(obj_id): #接受一个parse id (int)
    prompt = hico_obj_text_label[original_labels_dict[obj_id]][1].split()[4:]
    str = ""
    for word in prompt:
        if word == "a" or word == "an":
            break
        str = str + word + " "
    return "a " + str[:-1]
        
def get_prompt(v_o_list:list):
    race = random.choice(["asian", "black", "hispanic"])
    human = random.choice(["boy", "teenager", "man", "young man", "woman", "young woman"])
    quality = random.choice(["best quality","masterpiece"])
    details = random.choice(["Professional","Vivid Colors"])
    scene = random.choice(["spacious", "urban", "rustic"])
    shooting = random.choice(["DSLR","grainy","4K"])
    shooting2 = random.choice(["warm lighting","blue hour","backlit"])
    shooting3 = random.choice(["partial view","back view","front view"])
    shooting4 = random.choice(["Canon Eos5D","iphone 12"])
    #prompt_prefix = hico_text_label[v_o].replace("person",race+" "+human)+","
    hoi = "" 
    last_o = v_o_list[0][1]
    for v_o in v_o_list:
        if last_o != v_o[1]:
            hoi = hoi[:-4] + get_noun(last_o) + ","
        hoi = hoi + (get_verb(v_o)) + " and " 
        last_o = v_o[1]
    hoi = hoi[:-4] + get_noun(v_o_list[-1][1])
    prompt_prefix = "a photo of a " + race + " " + human + " " + hoi + ","  
    prompt_suffix = quality + "," + details + "," + scene + "," + \
    shooting + "," + shooting2 + "," + shooting3 + "," + shooting4
    return prompt_prefix + prompt_suffix
    
class SynPipeline:
    def __init__(self, config_path, ckpt_path) -> None:
        self.config_path = config_path
        self.model_checkpoint_path = ckpt_path

    def random_choice(self):  #随机选择要生成的vo元组列表
        # 首先 50-40-10 的选择从哪里选
        v_o_list = []
        seed = random.choice([0,0,0,0,0,1,1,1,1,2])  #50:multi 40:rare 10:non-rare
        if seed == 0:
            hois = random.choice(vo_pairs)
            for hoi in hois:
                v_o_list.append(id_to_hoi_dict[hoi])
            return v_o_list
        elif seed == 1:
            v_o_list.append(list(hico_text_label.keys())[random.choice(hico_unseen_index["rare_first"])])
        elif seed == 2:
            v_o_list.append(list(hico_text_label.keys())[random.choice(hico_unseen_index["non_rare_first"])])
        # 如果是只有一个（即不是multi抽的） 就按概率,根据multi变成多个（如果有）
        seed = random.choice([0,0,0,0,0,1])
        hoi_id = hoi_to_id_dict[v_o_list[0]] 
        if seed == 0 and hoi_id in multi_hoi:
            hois = [hoi for hoi in vo_pairs if hoi_id in hoi]
            #print(hois,hoi_id)
            while 1:
                hois_tuple = random.choice(hois)
                if hoi_id in hois_tuple:
                    break
            for hoi in hois_tuple:
                if hoi_id != hoi:
                    v_o_list.append(id_to_hoi_dict[hoi]) 
        return v_o_list
        
    # 1.生成图片
    def generate(self, pipe ,prompt):
        pipe.to("cuda")
        imgs = pipe(
            prompt,
            height=512,
            width=512,
            num_inference_steps=75,
            num_images_per_prompt=1,
            negative_prompt="mutated hands and fingers,poorly drawn hands,deformed,poorly drawn face,floating limbs,extra limb,floating limbs",
        ).images
        return imgs

    # 2.检测并过滤，然后  3.标注
    def detect_and_filter_and_anno(self, imgs, verbs_objs_tuple_list: list,out_dir,prompt):
        # == 读取json 为了之后的标注
        with open("./SynDatasets/annotations/train_val.json","r") as f:
            anno = json.load(f)
        f.close()
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
                    return
            # == 如果检测出来就做标注,并保存
            new_anno = generate_annotation(verbs_objs_tuple_list,tgt,formatted_name,files_num + 1) #先获得标注框
            if new_anno != None:
                with open("./SynDatasets/annotations/train_val.json","w") as f:
                    new_anno["prompt"] = prompt
                    img.save(out_dir + formatted_name)
                    anno.append(new_anno)
                    json.dump(anno, f, indent=4)
                f.close()
    
    # 自动化流程
    def run(self,SDpipe,imgs_num):
        for i in range(imgs_num):
            #v_o = random.choice(list(hico_text_label.keys())) #这个v_o是我改成原本了的 原先是(0开始的verb和预测的obj)
            v_o_list = self.random_choice()
            #print(v_o_list)
            prompt = get_prompt(v_o_list)  #找到对应提示词
            #print(prompt)
            imgs = pipeline.generate(SDpipe,prompt) 
            pipeline.detect_and_filter_and_anno(imgs,v_o_list,out_dir,prompt)
            print("目前进度:"+str(i+1)+"/"+str(imgs_num))

if __name__ == "__main__":
    out_dir = (
        "./SynDatasets/train_images/"
    )
    model_config_path = (
        "/root/autodl-tmp/DiffHOI/SynPipeline/DINO/config/DINO/DINO_4scale_swin.py"
    )
    model_checkpoint_path = (
        "/root/autodl-tmp/DiffHOI/params/checkpoint0011_4scale_swin.pth"
    )

    #==== pipeline
    SDpipe = StableDiffusionPipeline.from_pretrained(  #放在这是为了避免多次调用
        #"G:\数据集&权重\stable-diffusion-v1.5", torch_dtype=torch.float32
        "/root/autodl-tmp/DiffHOI/params/stable-diffusion-v1.5"
    )
    SDpipe = ""
    pipeline = SynPipeline(model_config_path, model_checkpoint_path)
    pipeline.run(SDpipe,100)
