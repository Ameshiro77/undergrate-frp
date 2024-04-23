# 本程序可视化并手动筛选

import cv2, os, sys, argparse
import json, random
import numpy as np
from labels_txt.labels import id_to_obj_dict, id_to_hoi_dict, id_to_verb_dict

# print(sys.path)
sys.path.append("./")

parser = argparse.ArgumentParser("Set output imgs num", add_help=False)
parser.add_argument("--start", default=1, type=int)


def click_corner(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(original_img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(
            original_img,
            xy,
            (x, y),
            cv2.FONT_HERSHEY_PLAIN,
            1.0,
            (0, 0, 0),
            thickness=1,
        )
        print(x, y)


def draw(img, tgt_dict):
    so_count = {}
    blk = np.zeros(img.shape, np.uint8)  # 透明背景用的
    # 画box
    boxes = tgt_dict["annotations"]  # 后面还有用
    for box_category in boxes:
        box = box_category["bbox"]
        category_id = box_category["category_id"]
        xy1 = (box[0], box[1])
        xy2 = (box[2], box[3])
        color = (np.random.random(3) * 0.6 + 0.4) * 255
        # print(color)
        cv2.rectangle(img, xy1, xy2, color, 2)  # 画人物的框
        text = id_to_obj_dict[category_id]
        cv2.rectangle(  # 用于text的背景框
            blk,
            (xy1[0], xy1[1]),
            (xy1[0] + len(text) * 10, xy1[1] + 25),
            color,
            -1,
        )
        cv2.putText(
            blk,
            text,
            (xy1[0], xy1[1] + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0, 0),
            2,
        )

    # 画交互
    for hoi in tgt_dict["hoi_annotation"]:
        xyxy1 = boxes[hoi["subject_id"]]["bbox"]  # 得到四个点
        xyxy2 = boxes[hoi["object_id"]]["bbox"]
        xy1 = (
            int(sum(xyxy1[0::2]) / 2),
            int(sum(xyxy1[1::2]) / 2),
        )  # 找到两个box的中点坐标
        xy2 = (int(sum(xyxy2[0::2]) / 2), int(sum(xyxy2[1::2]) / 2))
        # 如果是no interaction ， 画一条红线 ,无文字
        if hoi["category_id"] == 58:
            cv2.line(blk, xy1, xy2, (0, 0, 255), 3)
        else:
            s_o = (
                hoi["subject_id"],
                hoi["object_id"],
            )  # 记录主体客体对出现次数 调整text用的
            if so_count.get(s_o) == None:
                so_count[s_o] = 0
            else:
                so_count[s_o] = so_count[s_o] + 1
            text_pos = (
                int((xy1[0] + xy2[0]) / 2),
                int((xy1[1] + xy2[1]) / 2) + 28 * so_count[s_o],
            )  # 放文字的基准点
            text = id_to_verb_dict[hoi["category_id"]]
            cv2.rectangle(  # 用于text的背景框
                blk,
                (text_pos[0], text_pos[1]),
                (text_pos[0] + len(text) * 10, text_pos[1] + 25),
                (255, 255, 255),
                -1,
            )
            cv2.putText(
                blk,
                text,
                (text_pos[0], text_pos[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0, 0),
                2,
            )
            cv2.line(blk, xy1, xy2, (0, 255, 0), 3)
        # 画交互点
        cv2.circle(
            blk,
            (int((xy1[0] + xy2[0]) / 2), int((xy1[1] + xy2[1]) / 2)),
            3,
            (255, 0, 0),
            -1,
        )
        cv2.circle(
            blk,
            (xy1[0], xy1[1]),
            3,
            (255, 0, 0),
            -1,
        )
        cv2.circle(
            blk,
            (xy2[0], xy2[1]),
            3,
            (255, 0, 0),
            -1,
        )
    img = cv2.addWeighted(blk, 2, img, 0.85, 0)
    return img


# 此函数对过滤后的图片和标注重排序名称
def reorder_name(img_dir, json_path):
    with open(json_path, "r") as f:
        annotation = json.load(f)

    imgs = os.listdir(img_dir)

    for i, img_filename in enumerate(imgs):
        # 重命名
        src = os.path.join(img_dir, img_filename)
        new_name = img_filename[:10] + "{:06d}".format(i + 1) + ".jpg"
        tgt = os.path.join(img_dir, new_name)
        os.rename(src, tgt)
        # 重排标签
        annotation[i]["file_name"] = new_name
        annotation[i]["img_id"] = i + 1
    # 重写
    with open(json_path, "w") as f:
        json.dump(annotation, f, indent=2)


if __name__ == "__main__":
    is_syn = True
    args = parser.parse_args()
    if is_syn == True:
        img_dir = "./SynDatasets/train_images"
        json_path = "./SynDatasets/annotations/train_val.json"
    else:
        img_dir = r"G:\Code_Project\ComputerVision\no_frills_hoi_det-release_v1\HICO\hico_clean\hico_20160224_det\images\train2015"
        json_path = r"G:\Code_Project\ComputerVision\no_frills_hoi_det-release_v1\HICO\hico_clean\hico_20160224_det\annotations\trainval_hico.json"
    from_index = args.start  # 从第几个图片开始读★
    # == 先读取标注文件
    with open(json_path, "r") as f:
        annotation = json.load(f)

    # 将字典列表转换为字典，以便快速查找
    annotation_dict = {item["file_name"]: item for item in annotation}

    # == 遍历文件夹 依次读取图片
    imgs = os.listdir(img_dir)
    imgs_num = len(imgs)
    is_exit = False
    print("一共" + str(imgs_num) + "张图，标签共" + str(len(annotation)) + "个")
    if is_syn == True:
        assert len(annotation) == imgs_num
    low = [
        33,
        38,
        76,
        143,
        157,
        241,
        255,
        269,
        277,
        335,
        342,
        363,
        373,
        404,
        443,
        447,
        456,
        462,
        474,
        481,
        487,
        496,
    ]
    at_index = 0
    del_count = 0
    # 可视化，注意HICO DET数据集的标签比图片少
    for index, img_filename in enumerate(imgs):
        if index < from_index - 1:
            continue
        # if index not in low:
        #     continue
        # 显示图片
        target_dict = annotation_dict.get(img_filename)  # 得到了对应图片的标注字典
        if target_dict == None:  # 如果找不到图片就跳过,只针对hicodet数据集
            continue
        ctg = target_dict["annotations"]
        if ctg != None:
            objs_id = [ctg["category_id"] for ctg in target_dict["annotations"]]

        person_count = 0
        for id in objs_id:
            if id == 1:
                person_count = person_count + 1
        if person_count <= 1:
            continue
            
        if is_syn == False:
            hois = target_dict["hoi_annotation"]
            verb_ids = [_dict["category_id"] for _dict in hois]
            hoi_ids = [_dict["hoi_category_id"] for _dict in hois]
            # if 58 not in verb_ids:
            #     continue
            if 36 not in objs_id:
                continue
            # if 499 not in hoi_ids:
            #     continue

        prompt = target_dict.get("prompt")  # 名字打印到标题上
        if prompt == None:
            prompt = "example"
        else:
            prompt = prompt.split(",")[0] + "," + prompt.split(",")[1]
            
        img_path = os.path.join(img_dir, img_filename)
        original_img = cv2.imread(str(img_path))
        drwon_img = draw(original_img, target_dict)
        print(target_dict)
        print("当前图片:第" + str(index + 1) + "/" + str(imgs_num) + "张")
        # cv2.namedWindow(prompt)
        # cv2.setMouseCallback(prompt, click_corner)
        print(prompt)
        cv2.imshow(prompt, drwon_img)
        at_index = index + 1
        # 对图片进行操作：删除 保留 切换原图
        while 1:
            key = cv2.waitKey(0)
            if key == ord("q"):  # 退出
                is_exit = True
                break
            if key == ord("d") and is_syn == True:  # 删除文件
                os.remove(img_path)
                del_count = del_count + 1
                del annotation_dict[img_filename]
                break
            if key == ord("o"):  # 切换原图
                cv2.imshow(prompt, original_img)
                continue
            if key == ord("i"):  # 切换bbox图
                cv2.imshow(prompt, drwon_img)
                continue
            if key == 13 or key == 32 or key == 10:
                break
        if is_exit == True:
            break
        cv2.destroyAllWindows()

    # 删除完了后要重新进行写
    if is_syn == True:
        print("重写...")
        with open(json_path, "w") as f:
            new_annotation = list(annotation_dict.values())
            json.dump(new_annotation, f, indent=2)

    # 重排名序
    if is_syn == True:
        print("重排...")
        reorder_name(img_dir, json_path)

    # 在哪里退出
    imgs = os.listdir(img_dir)
    new_imgs_num = len(imgs)
    print("退出时的图片在当前位于:", new_imgs_num - (imgs_num - at_index))
