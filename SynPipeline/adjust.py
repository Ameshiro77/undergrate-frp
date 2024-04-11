# 调整一些标签用的
import json

img_dir = r"G:\Code_Project\ComputerVision\no_frills_hoi_det-release_v1\HICO\hico_clean\hico_20160224_det\images\train2015"
json_path = r"G:\Code_Project\ComputerVision\no_frills_hoi_det-release_v1\HICO\hico_clean\hico_20160224_det\annotations\trainval_hico.json"
with open(json_path, "r") as f:
    annotation = json.load(f)

new_anno = []
for anno in annotation:
    for hoi in anno["hoi_annotation"]:
        if hoi["hoi_category_id"] == 472:
            new_hoi = {}
            new_hoi["subject_id"] = hoi["subject_id"]
            new_hoi["object_id"] = hoi["object_id"]
            new_hoi["hoi_category_id"] = 473
            anno["hoi_annotation"].append(new_hoi)
    new_anno.append(anno)

with open(json_path,"w") as f:
    json.dump(new_anno, f, indent=2)
            