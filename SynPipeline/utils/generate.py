import os, sys,cv2
import argparse
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch, json, random
import numpy as np
sys.path.append("./")
import clip
from PIL import Image
from labels_txt.labels import id_to_hoi_dict, hoi_to_id_dict
from labels_txt.hico_text_label import hico_text_label
from labels_txt.vo_pairs import vo_pairs, multi_hoi
from utils.get_prompt import get_prompt


def get_hico_img(v_o_list):
    HICO_PATH = r"G:\Code_Project\ComputerVision\no_frills_hoi_det-release_v1\HICO\hico_clean\hico_20160224_det"
    if not os.path.exists(HICO_PATH):
        HICO_PATH = "/root/autodl-tmp/data/hico_20160224_det"
    json_path = os.path.join(HICO_PATH, "annotations", "trainval_hico.json")
    with open(json_path, "r") as f:
        annotation = json.load(f)
    name2ids_dict = {
        item["file_name"]: [
            hoi_item["hoi_category_id"] for hoi_item in item.get("hoi_annotation", [])
        ]
        for item in annotation
    }
    hoi_ids = [hoi_to_id_dict[vo] for vo in v_o_list] #给出的[hoi_id]列表
    found_dict = {k: v for k, v in name2ids_dict.items() if set(hoi_ids) <= set(v)} #寻找包含hois的HICO数据
    if len(found_dict.keys()) ==0:
        id = random.choice(hoi_ids)
        found_dict = {k: v for k, v in name2ids_dict.items() if id in v}  #如果找不到完全包含的，就随机选一个id找
    print(found_dict)
    img_name = random.choice(list(found_dict.keys()))  #随机挑一个满足条件的HICO数据
    img_path = os.path.join(HICO_PATH, "images","train2015", img_name)
    img = Image.open(img_path)
    # original_img = cv2.imread(str(img_path))
    # cv2.imshow("1", original_img)
    # cv2.waitKey(1)
    return img


def generate(pipe, v_o_list, steps, mode):
    prompt = get_prompt(v_o_list)
    ngt_prmt = "low quality,monochrome,skin blemishes,6 more fingers on one hand,deformity,bad legs,malformed limbs,extra limbs,ugly,poorly drawn hands,poorly drawn face,\
                                extra fingers,mutated hands,mutation,bad anatomy,disfigured,fused fingers,2 more person"
    if mode == "t2i":
        pipe.to("cuda")
        torch.cuda.empty_cache()
        imgs = pipe(
            prompt,
            height=512,
            width=512,
            num_inference_steps=steps,
            num_images_per_prompt=1,
            # negative_prompt="mutated hands and fingers,poorly drawn hands,deformed,poorly drawn face,floating limbs,low quality,",
            negative_prompt=ngt_prmt,
        ).images
        return imgs

    elif mode == "i2i":
        pipe.to("cuda")
        torch.cuda.empty_cache()
        imgs = pipe(
            prompt,
            image=get_hico_img(v_o_list),
            height=512,
            width=512,
            num_inference_steps=steps,
            num_images_per_prompt=1,
            # negative_prompt="mutated hands and fingers,poorly drawn hands,deformed,poorly drawn face,floating limbs,low quality,",
            negative_prompt=ngt_prmt,
        ).images
        return imgs


if __name__ == "__main__":
    # vos = [id_to_hoi_dict[i] for i in [224,225,226,227]]
    # print(get_hico_img(vos))
    # exit()
    SD_PATH = r"G:\数据集&权重\stable-diffusion-v1.5"
    # ==== SD pipeline
    gen = "i2i"
    if gen == "t2i":
        SDpipe = StableDiffusionPipeline.from_pretrained(  # 放在这是为了避免多次调用
            SD_PATH,
            # torch_dtype=torch.float32
        )
    elif gen == "i2i":
        SDpipe = StableDiffusionImg2ImgPipeline.from_pretrained(  # 放在这是为了避免多次调用
            SD_PATH,
            # torch_dtype=torch.float32
        )
    else:
        raise ValueError("生成方式不对,选择文生图t2i或图生图i2i")
    v_o_list = []
    hoi_id = [7, 8]
    for seq_hoi_id in hoi_id:
        v_o_list.append(id_to_hoi_dict[seq_hoi_id])
    imgs = generate(SDpipe, v_o_list, 50, gen)
    imgs.save("../example.jpg")
