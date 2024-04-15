# 用于合并数据集、合并标签

import os, json, shutil


def merge(source_dir, target_dir, hico_json_path, syn_json_path, target_json_path):
    imgs = os.listdir(source_dir)
    for index, img_filename in enumerate(imgs):
        source_file = os.path.join(source_dir, img_filename)
        target_file = os.path.join(target_dir, img_filename)
        shutil.copy(source_file, target_file)
    with open(hico_json_path, "r") as f:
        hico_annotation = json.load(f)
    with open(syn_json_path, "r") as f:
        syn_annotation = json.load(f)
    hico_annotation.extend(syn_annotation)
    with open(target_json_path, "w") as f:
        json.dump(hico_annotation, f, indent=2)

def delete_syn(hico_dir):
    imgs = os.listdir(hico_dir)
    for index, img_filename in enumerate(imgs):
        if "Syn" in img_filename:
            print(img_filename)
            target_file = os.path.join(hico_dir, img_filename)
            os.remove(target_file)
    


if __name__ == "__main__":
    HICO_PATH = r"G:\Code_Project\ComputerVision\no_frills_hoi_det-release_v1\HICO\hico_clean\hico_20160224_det"
    source_dir = r"./SynDatasets/train_images"
    target_dir = os.path.join(HICO_PATH, "images", "train2015")
    hico_json_path = os.path.join(HICO_PATH, "annotations", "trainval_hico.json")
    syn_json_path = r"./SynDatasets/annotations/train_val.json"
    target_json_path = os.path.join(HICO_PATH, "annotations", "new_trainval_hico.json")
    
    delete_syn(target_dir)
    merge(source_dir, target_dir, hico_json_path, syn_json_path, target_json_path)
