import json, os, torch
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional
from labels_txt.hico_text_label import hico_text_label, hico_unseen_index
from labels_txt.labels import id_to_hoi_dict
from tqdm import tqdm


def get_prompt(tgt):  # 获取[text]
    hois = tgt["hoi_annotation"]
    hoi_ids = [_dict["hoi_category_id"] for _dict in hois if _dict["category_id"] != 58]
    objs = [id_to_hoi_dict[id][1] for id in hoi_ids]
    print(objs)
    objs_num = len(set(objs))
    prompt_list = tgt["prompt"].split(",")[:objs_num]  # [x,x,..]
    prompt = ""
    for st in prompt_list:
        prompt = prompt + st
    return prompt


def analyse():
    # 用于分析HICO-DET数据集。
    img_dir = r"G:\Code_Project\ComputerVision\no_frills_hoi_det-release_v1\HICO\hico_clean\hico_20160224_det\images\train2015"
    json_path = r"G:\Code_Project\ComputerVision\no_frills_hoi_det-release_v1\HICO\hico_clean\hico_20160224_det\annotations\trainval_hico.json"
    with open(json_path, "r") as f:
        annotation = json.load(f)
        # 将字典列表转换为字典，以便快速查找
        annotation_dict = {item["file_name"]: item for item in annotation}

    hoi_count = {}
    verb_count = {}
    for i in range(600):
        hoi_count[i + 1] = 0
    for i in range(117):
        verb_count[i + 1] = 0
    # == 遍历文件夹 依次读取图片
    imgs = os.listdir(img_dir)
    imgs_num = len(imgs)

    for anno in annotation:
        for hoi in anno["hoi_annotation"]:
            hoi_id = hoi["hoi_category_id"]
            hoi_count[hoi_id] = hoi_count[hoi_id] + 1
            verb_id = hoi["category_id"]
            verb_count[verb_id] = verb_count[verb_id] + 1

    # 按键排序
    # hoi_count = {k: hoi_count[k] for k in sorted(hoi_count.keys())}

    # 按值排序
    hoi_count = dict(sorted(hoi_count.items(), key=lambda item: item[1]))
    verb_count = dict(sorted(verb_count.items(), key=lambda item: item[1]))
    # print(hoi_count)
    # print(verb_count)
    rare_hois = list(hoi_count.keys())[:600]

    # for idx in rare_hois:
    #     print(list(hico_text_label.values())[idx-1],hoi_count[idx])
    # print([hoi-1 for hoi in rare_hois])

    # 用于打印 按值排序的hoi三元组出现次数和索引（0开始）
    for index, i in enumerate([hoi - 1 for hoi in rare_hois]):
        print(index, i, hoi_count[i + 1])
    exit()
    print(sorted(hico_unseen_index["rare_first"]))
    plt.bar(range(600), list(hoi_count.values()))
    # plt.bar(range(117),list(verb_count.values()))
    # plt.show()


def clip_similarity():
    import clip, PIL

    img_dir = r"./SynDatasets/train_images"
    json_path = "./SynDatasets/annotations/train_val.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load("ViT-B/16", device=device)
    imgs = os.listdir(img_dir)
    imgs_num = len(imgs)
    cos_simi = []
    # 将字典列表转换为字典，以便快速查找
    with open(json_path, "r") as f:
        annotation = json.load(f)
    annotation_dict = {item["file_name"]: item for item in annotation}

    limit = 500
    low_simi = []
    low = [33, 38, 76, 143, 157, 241, 255, 269, 277, 335, 342, 363, 373, 404, 443, 447, 456, 462, 474, 481, 487, 496]
    with tqdm(total=limit) as pbar:
        with torch.no_grad():
            for index, img_filename in enumerate(imgs):
                if index == limit:
                    break
                # if index == 0:
                #     last_imgname = img_filename
                #     continue
                img = Image.open(os.path.join(img_dir, img_filename))
                prompt = get_prompt(annotation_dict[img_filename])
                #prompt = annotation_dict[img_filename]["prompt"]
                print(prompt)

                # 向量化
                image_embed = preprocess(img).unsqueeze(0).to(device)
                text_embed = clip.tokenize(prompt).to(device)
                # 提取特征
                image_features = model.encode_image(image_embed)  # 将图片进行编码  # [1,512]
                text_features = model.encode_text(text_embed)  # 将文本进行编码
                similarity = torch.nn.functional.cosine_similarity(image_features, text_features).item()
                cos_simi.append(similarity)
                if similarity < 0.25:
                    low_simi.append(index)
                #last_imgname = img_filename
                pbar.update(1)
            
    print(low_simi)
    plt.bar(range(limit), cos_simi)
    plt.show()
        


if __name__ == "__main__":
    clip_similarity()