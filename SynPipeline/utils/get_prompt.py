"""
本.py实现从一个[(v,o)..]列表生成prompt
"""
from labels_txt.labels import (
    id_to_verb_dict,
    id_to_obj_dict,
    valid_obj_ids,
    id_to_hoi_dict,
    original_labels_dict,
    hoi_to_id_dict,
)
from labels_txt.hico_text_label import (
    hico_text_label,
    hico_unseen_index,
    hico_obj_text_label,
)
from labels_txt.vo_pairs import vo_pairs, multi_hoi, aux_verb_noun
import random

# 获取动词
def get_verb(v_o):  # 接受一个(v,o)元组
    prompt = hico_text_label.get(v_o).split()[5:]
    str = ""
    for word in prompt:
        if word == "a" or word == "an":
            break
        str = str + word + " "
    return str[:-1]

# 获取名词
def get_noun(v_o):  # 接受一个(v,o)元组
    obj_id = v_o[1]
    prompt = hico_obj_text_label[original_labels_dict[obj_id]][1].split()[4:]
    str = ""
    for word in prompt:
        if word == "a" or word == "an":
            break
        str = str + word + " "
    return "a " + str[:-1]

# 获取全部的hoi短语
def get_hoi(v_o_list):
    hoi = ""
    last_vo = v_o_list[0]
    for v_o in v_o_list:
        if last_vo != v_o:
            hoi = hoi[:-4] + get_noun(last_vo) + aux_verb_noun[last_vo[0]] + ","
        hoi = hoi + (get_verb(v_o)) + " and "
        last_vo = v_o
    hoi = hoi[:-4] + get_noun(v_o_list[-1]) + aux_verb_noun[v_o_list[-1][0]]
    return hoi

# 获取提示词
def get_prompt(v_o_list: list):
    race = random.choice(["asian", "black", "hispanic"])
    human = random.choice(
        ["boy", "teenager", "man", "young man", "woman", "young woman"]
    )
    quality = random.choice(["best quality", "masterpiece"])
    details = random.choice(["Professional", "Vivid Colors"])
    # scene = random.choice(["spacious", "urban", "rustic"])
    scene = random.choice(["ultra realistic"])
    shooting = random.choice(["DSLR", "4K", "HD"])
    shooting2 = random.choice(["warm lighting", "blue hour"])
    # shooting3 = random.choice(["partial view","back view","front view"])
    shooting3 = random.choice(["Highly detailed"])
    shooting4 = random.choice(["Canon Eos5D", "iphone 12"])
    # prompt_prefix = hico_text_label[v_o].replace("person",race+" "+human)+","
    hoi = get_hoi(v_o_list)
    # (a photo of a asian young man cutting with a knife,cutting a carrot,holding a knife):1.05,
    prompt_prefix = "(a photo of a " + race + " " + human + " " + hoi + "):1.05,"
    prompt_suffix = (
        quality
        + ","
        + scene
        + ","
        + details
        + ","
        + shooting
        + ","
        + shooting2
        + ","
        + shooting3
        + ","
        + shooting4
    )
    return prompt_prefix + prompt_suffix