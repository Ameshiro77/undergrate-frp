from diffusers import DiffusionPipeline,StableDiffusionPipeline
import os, sys
import torch, json
import numpy as np
from PIL import Image
sys.path.append("./DINO")
import DINO.datasets.transforms as T
from DINO.main import build_model_main
from DINO.util.slconfig import SLConfig
from DINO.datasets import build_dataset
from DINO.util.visualizer import COCOVisualizer
from DINO.util import box_ops

#1. 生成图片
def generate_img(prompt,out_name):
    pipe = DiffusionPipeline.from_pretrained(r"/root/autodl-tmp/DiffHOI/params/stable-diffusion-v1.5/", torch_dtype=torch.float32)
    pipe.to("cuda")
    pipe(prompt).images[0].save(out_name)

