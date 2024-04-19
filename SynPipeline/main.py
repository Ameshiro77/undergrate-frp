import os, sys
import argparse
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch, json, random
import numpy as np
import clip
from PIL import Image
from labels_txt.labels import id_to_hoi_dict, hoi_to_id_dict
from labels_txt.hico_text_label import hico_text_label
from labels_txt.vo_pairs import vo_pairs, multi_hoi

sys.path.append("./DINO")
from utils.get_prompt import get_prompt
from utils.detect import detect
from utils.anno_json import generate_annotation
from labels_txt.rare_list import rare_list

parser = argparse.ArgumentParser("Set output imgs num", add_help=False)
parser.add_argument("--nums", default=1, type=int)
parser.add_argument("--steps", default=75, type=int)
parser.add_argument("--rare", default=160, type=int)  # 表明前多少个算rare
parser.add_argument("--mode", default="random", type=str)  # random | seq
parser.add_argument("--gen", default="t2i", type=str)  # t2i | i2i


class SynPipeline:
    def __init__(self, config_path, ckpt_path) -> None:
        self.config_path = config_path
        self.model_checkpoint_path = ckpt_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)

    """
    # 随机选择要生成的vo元组列表
    """
    def random_choice(self, start, mode, seq_hoi_id=None):  # 随机选择要生成的vo元组列表
        v_o_list = []
        if mode == "random":  # 如果是random模式，就随机按权重抽取
            # 首先选择从哪里选
            seed = random.choice([1, 1, 1, 1, 1, 1, 1, 2, 2, 2])  # 70:rare 30:non-rare
            if seed == 0:
                hois = random.choice(vo_pairs)
                for hoi in hois:
                    v_o_list.append(id_to_hoi_dict[hoi])
                return v_o_list
            elif seed == 1:
                v_o_list.append(
                    list(hico_text_label.keys())[random.choice(rare_list[:start])]
                )
            elif seed == 2:
                v_o_list.append(
                    list(hico_text_label.keys())[random.choice(rare_list[start:])]
                )
            elif seed == 3:
                v_o_list.append(list(hico_text_label.keys())[random.randint(0, 599)])
            hoi_id = hoi_to_id_dict[v_o_list[0]]
        elif mode == "seq":  # 如果是顺序模式，就顺序按rare程度以此生成图片
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

    """
    1.生成图片
    """
    def generate(self, pipe, prompt, steps, mode):
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
                height=512,
                width=512,
                num_inference_steps=steps,
                num_images_per_prompt=1,
                # negative_prompt="mutated hands and fingers,poorly drawn hands,deformed,poorly drawn face,floating limbs,low quality,",
                negative_prompt=ngt_prmt,
            ).images
            return imgs

    """
    2.检测并过滤，然后  3.标注
    """
    def detect_and_filter_and_anno(
        self, imgs, verbs_objs_tuple_list: list, out_dir, prompt
    ):
        # == 读取json 为了之后的标注
        with open("./SynDatasets/annotations/train_val.json", "r") as f:
            anno = json.load(f)
        f.close()
        # file_name,id
        files_num = anno[-1]["img_id"]
        formatted_name = "Syn_train_" + "{:06d}".format(files_num + 1) + ".jpg"
        # == 对每个图像检测
        for img in imgs:
            # ========= 首先，用CLIP过滤
            with torch.no_grad():
                image_embed = (
                    self.preprocess(img).unsqueeze(0).to(self.device)
                )  # 向量化
                text_embed = clip.tokenize(prompt).to(self.device)
                image_features = self.model.encode_image(
                    image_embed
                )  # 将图片进行编码  # [1,512]
                text_features = self.model.encode_text(text_embed)  # 将文本进行编码
                similarity = torch.nn.functional.cosine_similarity(
                    image_features, text_features
                ).item()
                if similarity < 0.25:
                    print("CLIP检测不通过")
                    return
            # ===== 然后检测目标，过滤，标注
            tgt = detect(img, self.config_path, self.model_checkpoint_path)
            # 检查是否检测出了指定的物体
            for v_o in verbs_objs_tuple_list:
                obj_id = v_o[1]
                if obj_id not in tgt["box_label_parse_id"]:
                    # print(tgt)
                    # raise ValueError("错误！没检测出生成图片时指定的物体！")
                    print("没检测出指定物体")
                    return
            # == 如果检测出来就做标注,并保存
            new_anno = generate_annotation(
                verbs_objs_tuple_list, tgt, formatted_name, files_num + 1
            )  # 先获得标注框
            if new_anno != None:
                with open("./SynDatasets/annotations/train_val.json", "w") as f:
                    new_anno["prompt"] = prompt
                    img.save(out_dir + formatted_name)
                    print("保存图片:", formatted_name)
                    anno.append(new_anno)
                    json.dump(anno, f, indent=2)
                f.close()

    """
    自动化流程
    """
    # def run(self, SDpipe, imgs_num, rare_num, mode, steps):
    def run(self, *args):
        # v_o_list = self.random_choice(rare_num, mode)
        # print(get_prompt(v_o_list))  # 找到对应提示词
        # exit()
        # ==== 随机按权重从600个hoi里挑选生成
        if args.mode == "random":
            for i in range(args.imgs_num):
                # v_o = random.choice(list(hico_text_label.keys())) #这个v_o是我改成原本了的 原先是(0开始的verb和预测的obj)
                v_o_list = self.random_choice(args.rare_num, args.mode)
                prompt = get_prompt(v_o_list)  # 找到对应提示词
                imgs = pipeline.generate(SDpipe, prompt, args.steps, args.gen)
                pipeline.detect_and_filter_and_anno(imgs, v_o_list, out_dir, prompt)
                print("目前进度:" + str(i + 1) + "/" + str(args.imgs_num))
        # ==== 随机从尾部N个类别里依次生成M个
        if args.mode == "seq":
            count = 0
            from analyse import get_rare_list

            HICO_PATH = "/root/autodl-tmp/data/hico_20160224_det"
            _rare_list = get_rare_list(HICO_PATH, args.rare)
            random.shuffle(_rare_list)
            sum = args.rare_num * args.imgs_num
            for i in range(args.rare_num):
                for j in range(args.imgs_num):
                    count = count + 1
                    vo = id_to_hoi_dict[_rare_list[i]]
                    if vo[0] == 58 or vo[1] == 1:
                        print("无交互或对象是人")
                        continue
                    v_o_list = self.random_choice(
                        args.rare_num, args.mode, _rare_list[i]
                    )
                    prompt = get_prompt(v_o_list)
                    imgs = pipeline.generate(SDpipe, prompt, args.steps, args.gen)
                    pipeline.detect_and_filter_and_anno(imgs, v_o_list, out_dir, prompt)
                    # count = count + 1
                    print("目前进度:" + str(count) + "/" + str(sum))


if __name__ == "__main__":
    out_dir = "./SynDatasets/train_images/"
    model_config_path = (
        "/root/autodl-tmp/frp/SynPipeline/DINO/config/DINO/DINO_4scale_swin.py"
    )
    model_checkpoint_path = "/root/autodl-tmp/frp/params/checkpoint0011_4scale_swin.pth"

    args = parser.parse_args()
    # ==== pipeline
    SDpipe = None
    if args.gen == "t2i":
        SDpipe = StableDiffusionPipeline.from_pretrained(  # 放在这是为了避免多次调用
            # "G:\数据集&权重\stable-diffusion-v1.5", torch_dtype=torch.float32
            "/root/autodl-tmp/frp/params/stable-diffusion-v1.5"
        )
    elif args.gen == "i2i":
        SDpipe = StableDiffusionImg2ImgPipeline.from_pretrained(  # 放在这是为了避免多次调用
            # "G:\数据集&权重\stable-diffusion-v1.5", torch_dtype=torch.float32
            "/root/autodl-tmp/frp/params/stable-diffusion-v1.5"
        )
    else:
        raise ValueError("生成方式不对,选择文生图t2i或图生图i2i")

    from diffusers import StableDiffusionImg2ImgPipeline

    pipeline = SynPipeline(model_config_path, model_checkpoint_path)
    pipeline.run(SDpipe, args)
