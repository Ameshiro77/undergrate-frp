import json, os
import matplotlib.pyplot as plt
from labels_txt.hico_text_label import hico_text_label,hico_unseen_index
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
    hoi_count[i+1] = 0
for i in range(117):
    verb_count[i+1] = 0
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
#hoi_count = {k: hoi_count[k] for k in sorted(hoi_count.keys())}

# 按值排序
hoi_count = dict(sorted(hoi_count.items(), key= lambda item : item[1]))
verb_count = dict(sorted(verb_count.items(), key= lambda item : item[1]))
#print(hoi_count)
#print(verb_count)
rare_hois = list(hoi_count.keys())[:600]

# for idx in rare_hois:
#     print(list(hico_text_label.values())[idx-1],hoi_count[idx])
#print([hoi-1 for hoi in rare_hois])

# 用于打印 按值排序的hoi三元组出现次数和索引（0开始）
for index,i in enumerate([hoi-1 for hoi in rare_hois]):
    print(index,i,hoi_count[i+1])
exit()
print(sorted(hico_unseen_index["rare_first"]))
plt.bar(range(600),list(hoi_count.values()))
# plt.bar(range(117),list(verb_count.values()))
# plt.show()
