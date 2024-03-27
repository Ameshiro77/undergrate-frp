import sys
sys.path.append("./")
# 用于生成各种id对应关系。
from labels_txt.labels import valid_obj_ids,valid_verb_ids,verb_to_id_dict
from labels_txt.labels import hoi_to_id_dict,obj_to_id_dict,original_labels_dict,verb_to_id_dict

# ===== id 与 verb ======
# 根据hoi_verb_id  获取verb to id的字典  （1~117）
def get_verb_to_id_dict():
    verb_to_id_dict = {}
    with open("./labels_txt/hico_list_vb.txt", "r") as f:
        id_verbs = f.readlines()[2:]
        for label in id_verbs:
            list = label.split()
            verb_to_id_dict[list[1]] = int(list[0])
    f.close()
    #print(verb_to_id_dict)
    return verb_to_id_dict


# 获取id to verb
def get_id_to_verb_dict():
    return inverse_dict(get_verb_to_id_dict())


# ===== id 与 obj =====
# 根据obj_id 获取obj to id的字典 （注意，这里是到90的解析序号）
def get_obj_to_id_dict():
    from labels_txt.labels import valid_obj_ids
    obj_to_id_dict = {}
    import json
    with open("labels_txt/coco_id2name.json","r") as f:
        id2name = json.load(f)
    id2name = {int(k):v for k,v in id2name.items()}
    obj_to_id_dict = inverse_dict(id2name)    
    f.close()
    return obj_to_id_dict
    
    # 注释掉的是根据hoi_obj.txt转的，这个是错的！
    # with open("./labels_txt/hico_list_obj.txt", "r") as f:
    #     id_objs = f.readlines()[2:]
    #     for label in id_objs:
    #         list = label.split()
    #         obj_to_id_dict[list[1]] = valid_obj_ids[int(list[0]) - 1]

    # f.close()
    # return obj_to_id_dict

def get_id_to_obj_dict():
    return inverse_dict(get_obj_to_id_dict())


# ==========
# 建立一个元组(verb_id obj_id)：hoi_id的关系，而不用字符串。
def get_hoi_to_id_dict():
    hoi_to_id_dict = {}
    with open("./labels_txt/hico_list_hoi.txt", "r") as f:
        id_hois = f.readlines()[2:]
        verb_to_id_dict = get_verb_to_id_dict()
        obj_to_id_dict = get_obj_to_id_dict()
        #print(obj_to_id_dict)
        for label in id_hois:
            list = label.split()
            #print(list)
            v_o = (verb_to_id_dict[list[2]],obj_to_id_dict[list[1]])
            hoi_to_id_dict[v_o] = int(list[0])
    f.close()
    return hoi_to_id_dict

'''
如果找不到就返回None
'''
# 根据verb obj的解析id 获取对应的hoi标签id
def get_hoi_id(v_o:tuple):
    return hoi_to_id_dict.get(v_o)


# == misc
def inverse_dict(dict):
    inverse_dict = {v: k for k, v in dict.items()}
    return inverse_dict


# 根据obj_id元组 获得解析：预测字典
def get_parse_to_original_labels_dict():
    original_labels = {}
    for i, j in enumerate(valid_obj_ids):
        original_labels[j] = i
    return original_labels


if __name__ == "__main__":
    #print(get_obj_to_id_dict())
    print(get_hoi_to_id_dict())
    pass


