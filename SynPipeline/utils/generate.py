from diffusers import DiffusionPipeline, StableDiffusionPipeline
import os, sys
import torch, json
import numpy as np
from PIL import Image


# 1. 生成图片 out_name带.jpg
def generate_img(prompt):
    # pipe = DiffusionPipeline.from_pretrained(r"/root/autodl-tmp/DiffHOI/params/stable-diffusion-v1.5/", torch_dtype=torch.float32)
    pipe = StableDiffusionPipeline.from_pretrained(
        #r"G:\数据集&权重\stable-diffusion-v1.5", torch_dtype=torch.float32
        "/root/autodl-tmp/DiffHOI/params/stable-diffusion-v1.5"
    )
    pipe.to("cuda")
    imgs = pipe(
        prompt,
        height=512,
        width=512,
        num_inference_steps=75,
        negative_prompt="mutated hands and fingers,poorly drawn hands,deformed,poorly drawn face,floating limbs,extra limb,floating limbs",
    ).images
    return imgs


if __name__ == "__main__":
    prompt = "Photo of a young girl smiling,urban,4K,backlit,partial view,ACG,comic"
    out_name = "test.jpg"
    generate_img(prompt, out_name)
