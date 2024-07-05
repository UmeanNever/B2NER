import json
import os
import csv
from tqdm import tqdm
import time
import random

# 逐个读取数据集的数据
def get_dataset(dataset_dir):
    with open(os.path.join(dataset_dir,"train.json"),"r",encoding = "utf-8") as f:
        datasets = json.loads(f.read())
    return datasets

def in_sentence(entity,sentence):
    entity = entity.lower()
    sentence = sentence.lower()
    if " "+entity+" " in  sentence or sentence.startswith(entity+" ") or sentence.endswith(" "+entity):
        return True
    
    return False

def inside_entities(entity,full_entities):
    for full_entity in full_entities:
        if in_sentence(entity, full_entity):
            return True
    return False

# entity是重合标签名，datas是待匹配的sentence,entities是待匹配的所有抽取实体                    
def match(entity,datas,entities):
    entity = entity.lower()
    count_matched = 0
    not_extract = []
    wrong_extract = []
    span_extract = []
    for entity_extracted in tqdm(entities):
        # 在所有的数据里面匹配
        for data in datas:
            sentence = data["sentence"].lower()
            entity_extracted = entity_extracted.lower()
            if not in_sentence(entity_extracted,sentence):
                continue
            data_entity_set_total = set([entity_["name"].lower() for entity_ in data["entities"]])
            data_entity_set_other = set([entity_["name"].lower() for entity_ in data["entities"] if entity_["type"].lower()!=entity])
            data_entity_set_same = set([entity_["name"].lower() for entity_ in data["entities"] if entity_["type"].lower()==entity])
            if not inside_entities(entity_extracted,data_entity_set_total):
                count_matched += 1
                if entity_extracted in data_entity_set_same:
                    break
                elif entity_extracted in data_entity_set_other:
                    wrong_extract.append((entity_extracted,sentence))
                    break
                else:
                    part = [i.lower() for i in data_entity_set_same if in_sentence(i.lower(), entity_extracted)]
                    if len(part)==0:
                        not_extract.append((entity_extracted,sentence))
                    else:
                        span_extract.append((entity_extracted,part[0],sentence))
                    break
    return count_matched, not_extract, wrong_extract, span_extract
    


def same_content(datasets,data_dict):
    write_content = []
    entity_dict = {}
    # 统计每个类别entity的集合
    start_time = time.time()
    for key in data_dict:
        datas = data_dict[key]
        entity_dict[key] = {}
        for data in datas:
            for entity in data["entities"]:
                if entity["type"] in entity_dict[key]:
                    entity_dict[key][entity["type"]].append(entity["name"])
                else:
                    entity_dict[key][entity["type"]] = [entity["name"]]
    print("统计每个标签的集合耗时",time.time()-start_time)
    # 同名标签对一共有388对（在统一大小写以后），那么正反向分别比对，需要重复776次
    for i in range(len(datasets)):
        for j in range(i+1,len(datasets)):
            # 遍历所有数据集对
            dataset1,dataset2 = datasets[i],datasets[j]
            entity_set1,entity_set2 = entity_dict[dataset1].keys(),entity_dict[dataset2].keys()
            datas1,datas2 = data_dict[dataset1],data_dict[dataset2]
            for entity1 in entity_set1:
                for entity2 in entity_set2:
                    if entity1.lower()!=entity2.lower():
                        continue
                    
                    # 首先判断第一个数据集对第二数据集
                    random.shuffle(datas2)
                    random.shuffle(datas1)
                    
                    entities1 = entity_dict[dataset1][entity1]
                    random.shuffle(entities1)
                    entities1 = entities1[:min(2000,len(entities1))]
                    
                    entities2 = entity_dict[dataset2][entity2]
                    random.shuffle(entities2)
                    entities2 = entities2[:min(2000,len(entities2))]
                    
                    count_matched1, not_extract1, wrong_extract1, span_extract1 = match(entity1,datas2[:min(len(datas2),2000)],entities1)
                    ratio1 = (len(not_extract1)+len(wrong_extract1)+len(span_extract1))/count_matched1 if count_matched1!=0 else 0
                    
                    count_matched2, not_extract2, wrong_extract2, span_extract2 = match(entity2,datas1[:min(len(datas1),2000)],entities2)
                    ratio2 = (len(not_extract2)+len(wrong_extract2)+len(span_extract2))/count_matched2 if count_matched2!=0 else 0
                    
                    write_content.append((ratio1, dataset1, dataset2, entity1, count_matched1,
                                          len(not_extract1), len(wrong_extract1), len(span_extract1),
                                          not_extract1[:min(20, len(not_extract1))],
                                          wrong_extract1[:min(20, len(wrong_extract1))],
                                          span_extract1[:min(20, len(span_extract1))]))
                    write_content.append((ratio2, dataset2, dataset1, entity2, count_matched2,
                                          len(not_extract2), len(wrong_extract2), len(span_extract2),
                                          not_extract2[:min(20, len(not_extract2))],
                                          wrong_extract2[:min(20, len(wrong_extract2))],
                                          span_extract2[:min(20, len(span_extract2))]))

    # Writing to file
    with open("conflict3.tsv", "w", encoding="utf-8") as f:
        write_content.sort(reverse=True)
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=["dataset_entity", "match_dataset", "len_entities_found", "len_not_extracted", "len_wrong_extracted", "len_span_extracted",
                                                               "ratio", "not_match_examples", "wrong_match_examples", "span_match_examples"])
        writer.writeheader()
        for line in write_content:
            writer.writerow({
                "dataset_entity": f"{line[1]}->{line[3]}",
                "match_dataset": line[2],
                "len_entities_found": line[4],
                "len_not_extracted": line[5],
                "len_wrong_extracted": line[6],
                "len_span_extracted": line[7],
                "ratio": line[0],
                "not_match_examples": str(line[8]),
                "wrong_match_examples": str(line[9]),
                "span_match_examples": str(line[10])
            })
    

if __name__ == "__main__":
    root_dir = "/mnt/data/user/yang_yuming/data/Public/B2NERD/NER_en"
    datasets = os.listdir(root_dir)
    # 已有采样版本的，针对采样的版本进行统计
    for dataset in set(datasets):
        if "sample" in dataset:
            datasets.remove(dataset)
    
    data_dict = {}
    start_time = time.time()
    for dataset in datasets:
        datas = get_dataset(os.path.join(root_dir,dataset))
        # 用一个字典来存储所有的数据
        data_dict[dataset] = datas
    print("加载数据耗时：",time.time()-start_time)
    
    same_content(datasets,data_dict)
    
    