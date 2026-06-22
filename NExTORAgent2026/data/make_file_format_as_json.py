import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
dataset_name = "complexor.jsonl"
# Path to the JSON file


def make_file_format_as_json(dataset_name):
    json_file_path = f"LLMOPTdataset/{dataset_name}"
    data = {}
    with open(json_file_path, 'r', encoding='utf-8') as file:
        ind = 0
        for line in file:
            line = line.strip()
            obj = json.loads(line)
            data[str(ind)] = obj
            ind += 1


    first = dataset_name.split(".")[0]
    last = dataset_name.split(".")[1]
    dataset_name = first+"."+last.replace("l","")
    print(dataset_name)
    # Extract question lengths
    with open(f"NExT_datasets/{dataset_name}", 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return data


def get_num_nonelin(json_file_path):

    with open(f"{json_file_path}.json", 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    # 用列表推导：把所有符合条件的记录放进新列表，再计数
    filtered_list_NLP = []
    filtered_list = []
    for item in data_list:
        if isinstance(item, dict) and item.get("type") == "nonlinear-notable":
            filtered_list_NLP.append(item)
        else:
            filtered_list.append(item)

    data = {}
    data_NLP = {}
    i = 0
    j = 0
    for item in filtered_list:
        item["answer"] = [float(re[-1]) for re in item["results"].items()]
        data[str(i)] = item
        i += 1
    for item in filtered_list_NLP:
        item["answer"] = [float(re[-1]) for re in item["results"].items()]
        data_NLP[str(j)] = item
        j = j + 1


    with open(f"{json_file_path}_NLP.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    with open(f"{json_file_path}_LP.json", 'w', encoding='utf-8') as f:
        json.dump(data_NLP, f, ensure_ascii=False, indent=4)

    print(f"共找到 {len(filtered_list)} 条 “type = 'nonlinear-notable'” 的记录")
    return data
    # 如果需要，也可以打印 filtered_list 里的内容
    # print(filtered_list)
if __name__ == '__main__':
    get_num_nonelin("origin_data/OptiBench")