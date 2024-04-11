# 调整一些标签用的
import json

img_dir = "./SynDatasets/train_images"
json_path = "./SynDatasets/annotations/train_val.json"
with open(json_path, "r") as f:
    annotation = json.load(f)

new_anno = []
for anno in annotation:
    for hoi in anno["hoi_annotation"]:
        if hoi["hoi_category_id"] == 481:
            new_hoi = {}
            new_hoi["subject_id"] = hoi["subject_id"]
            new_hoi["object_id"] = hoi["object_id"]
            new_hoi["category_id"] = 115
            new_hoi["hoi_category_id"] = 482
            anno["hoi_annotation"].append(new_hoi)
    new_anno.append(anno)

with open(json_path,"w") as f:
    json.dump(new_anno, f, indent=2)
            