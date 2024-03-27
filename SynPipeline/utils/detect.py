#from diffusers import DiffusionPipeline,StableDiffusionPipeline
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

"""
2.检测人-物框
"""
def detect(img_path,config_path,checkpoint_path) -> dict:
    model_config_path = config_path # change the path of the model config file
    #model_checkpoint_path = r"G:\数据集&权重\checkpoint0031_5scale.pth"  # change the path of the model checkpoint
    model_checkpoint_path = checkpoint_path

    args = SLConfig.fromfile(model_config_path) 
    args.device = 'cuda' 
    model, criterion, postprocessors = build_model_main(args)
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    _ = model.eval()

    # load coco names
    with open('./DINO/util/coco_id2name.json') as f:
        id2name = json.load(f)
        id2name = {int(k):v for k,v in id2name.items()}

    image = Image.open(img_path).convert("RGB") # load image
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image, _ = transform(image, None)
    output = model.cuda()(image[None].cuda())
    output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
    thershold = 0.5 # set a thershold

    vslzr = COCOVisualizer()

    scores = output['scores']
    labels = output['labels']
    boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
    select_mask = scores > thershold

    from utils.labels_dict import original_labels_dict
    box_label_parse_id = [int(item) for item in labels[select_mask] ]
    box_label = [id2name[int(item)] for item in labels[select_mask]]
    pred_dict = {
        'boxes': boxes[select_mask],
        'size': torch.Tensor([image.shape[1], image.shape[2]]),
        'box_label': box_label,
        'box_label_parse_id':box_label_parse_id
    }
    #print(pred_dict)
    vslzr.visualize(image, pred_dict, savedir=None, dpi=100)  #保存图片
    return pred_dict

if __name__ == "__main__":
    img_path = "./output.jpg"
    config_path = r"/root/autodl-tmp/DiffHOI/SynPipeline/DINO/config/DINO/DINO_4scale_swin.py"
    model_checkpoint_path =  r"/root/autodl-tmp/DiffHOI/params/checkpoint0011_4scale_swin.pth"
    detect(img_path,config_path,model_checkpoint_path)