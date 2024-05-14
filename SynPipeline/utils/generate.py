import os, sys, cv2
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


def random_choice(HICO_PATH):  # 随机选择要生成的vo元组列表
    from analyse import get_rare_list
    PATH = HICO_PATH
    _rare_list = get_rare_list(PATH, 160)
    random.shuffle(_rare_list)
    seq_hoi_id = random.choice(_rare_list)
    v_o_list = []
    v_o_list.append(id_to_hoi_dict[seq_hoi_id])
    hoi_id = seq_hoi_id

    # 按概率,根据multi变成多个（如果有）
    seed = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    if seed == 0 and hoi_id in multi_hoi:  # 补全可能的多标签
        hois = [hoi for hoi in vo_pairs if hoi_id in hoi]
        if len(hois) == 0:
            return v_o_list
        # print(hois,hoi_id)
        while 1:
            hois_tuple = random.choice(hois)
            if hoi_id in hois_tuple:
                break
        for hoi in hois_tuple:
            if hoi_id != hoi:
                v_o_list.append(id_to_hoi_dict[hoi])
    return v_o_list


def get_name2ids_dict(HICO_PATH):
    print(HICO_PATH)
    json_path = os.path.join(HICO_PATH, "annotations", "trainval_hico.json")
    with open(json_path, "r") as f:
        annotation = json.load(f)
    imgname2hoiids_dict = {
        item["file_name"]: [
            hoi_item["hoi_category_id"] for hoi_item in item.get("hoi_annotation", [])
        ]
        for item in annotation
    }
    return imgname2hoiids_dict


def get_hico_img(v_o_list, HICO_PATH):
    name2ids_dict = get_name2ids_dict(HICO_PATH)
    hoi_ids = [hoi_to_id_dict[vo] for vo in v_o_list]  # 给出的[hoi_id]列表
    found_dict = {
        k: v for k, v in name2ids_dict.items() if set(hoi_ids) <= set(v)
    }  # 寻找包含hois的HICO数据
    if len(found_dict.keys()) == 0:
        id = random.choice(hoi_ids)
        found_dict = {
            k: v for k, v in name2ids_dict.items() if id in v
        }  # 如果找不到完全包含的，就随机选一个id找
    # print(found_dict)
    img_name = random.choice(list(found_dict.keys()))  # 随机挑一个满足条件的HICO数据
    print(img_name, name2ids_dict[img_name])
    img_path = os.path.join(HICO_PATH, "images", "train2015", img_name)
    img = Image.open(img_path)
    img.save("./original.jpg")
    return img


def generate(pipe, v_o_list, steps, mode, HICO_PATH):
    prompt = get_prompt(v_o_list)
    print(prompt)
    ngt_prmt = "monochrome,skin blemishes,6 more fingers on one hand,deformity,bad legs,malformed limbs,extra limbs,ugly,poorly drawn hands,poorly drawn face,\
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
            negative_prompt=ngt_prmt,
        ).images
        return imgs

    elif mode == "i2i":
        pipe.to("cuda")
        torch.cuda.empty_cache()
        imgs = pipe(
            prompt,
            image=get_hico_img(v_o_list, HICO_PATH),
            height=512,
            width=512,
            strength=0.83,
            guidance_scale=7.65,
            num_inference_steps=steps,
            num_images_per_prompt=1,
            negative_prompt=ngt_prmt,
        ).images
        return imgs


if __name__ == "__main__":
    # vos = [id_to_hoi_dict[i] for i in [224,225,226,227]]
    # print(get_hico_img(vos))
    # exit()
    HICO_PATH = r"G:\Code_Project\ComputerVision\no_frills_hoi_det-release_v1\HICO\hico_clean\hico_20160224_det"
    if not os.path.exists(HICO_PATH):
        HICO_PATH = "/root/autodl-tmp/data/hico_20160224_det"

    SD_PATH = r"G:\数据集&权重\stable-diffusion-v1.5"
    if not os.path.exists(SD_PATH):
        SD_PATH = "/root/autodl-tmp/frp/params/stable-diffusion-v1.5/"

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
    # from diffusers import DDPMScheduler

    # SDpipe.scheduler = DDPMScheduler.from_config(SDpipe.scheduler.config)
    print(SDpipe.config)
    #     exit()
    v_o_list = random_choice(HICO_PATH)
    # hoi_id = (386, 388)
    # for seq_hoi_id in hoi_id:
    #     v_o_list.append(id_to_hoi_dict[seq_hoi_id])
    print(v_o_list)
    # imgs = generate(SDpipe, v_o_list, 80, gen, HICO_PATH)
    # imgs[0].save("./example.jpg")
