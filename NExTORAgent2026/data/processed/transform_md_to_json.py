import json
import re


def convert_md_to_json(input_filename):
    # 打开并读取 Markdown 文件
    with open(input_filename, 'r', encoding='utf-8') as file:
        md_content = file.read()

    # 使用正则表达式提取所有 prob_XXX 标题及其内容
    prob_pattern = re.compile(r'# (prob_[^\n]+)\n([\s\S]+?)(?=\n# prob_[^\n]+|\Z)', re.MULTILINE)

    # 存储所有 prob_xxx 的结果
    prob_list = []

    # 遍历每个 prob_xxx 的内容
    for match in prob_pattern.finditer(md_content):
        prob_title = match.group(1).strip()  # 一级标题
        prob_content = match.group(2).strip()  # 该 prob_xxx 下的所有内容

        # 判断是否带有❌️，如果是则跳过
        if '❌️' in prob_title:
            continue

        # 提取具体的数据
        question = None
        answer = "None"
        index = None
        NLtype = None
        NLChange = None
        model_path = None
        code_path = None

        # 提取 question 内容
        question_match = re.search(r'## question\n([\s\S]+?)(?=\n###|$)', prob_content)
        if question_match:
            question = question_match.group(1).strip().replace('\n', '\\n')

        # 提取 answer 内容
        answer_match = re.search(r'- \*\*answer\*\*: ([^\n]+)', prob_content)
        if answer_match:
            answer_text = answer_match.group(1).strip().lower()
            if answer_text in ["none", "none"]:
                answer = "None"
            else:
                try:
                    answer = float(answer_text)
                except ValueError:
                    answer = "None"

        # 提取其他字段
        index_match = re.search(r'- \*\*index\*\*: ([^\n]+)', prob_content)
        if index_match:
            index = index_match.group(1).strip()

        NLtype_match = re.search(r'- \*\*NLtype\*\*: ([^\n]+)', prob_content)
        if NLtype_match:
            NLtype = NLtype_match.group(1).strip()

        NLChange_match = re.search(r'- \*\*NLChange\*\*: ([^\n]+)', prob_content)
        if NLChange_match:
            NLChange = NLChange_match.group(1).strip()

        model_path_match = re.search(r'- \*\*model_path\*\*: ([^\n]+)', prob_content)
        if model_path_match:
            model_path = model_path_match.group(1).strip()

        code_path_match = re.search(r'- \*\*code_path\*\*: ([^\n]+)', prob_content)
        if code_path_match:
            code_path = code_path_match.group(1).strip()

        # 添加到列表中
        prob_list.append({
            "title": prob_title,
            "question": question,
            "answer": answer,
            "index": index,
            "NLtype": NLtype,
            "NLChange": NLChange,
            "model_path": model_path,
            "code_path": code_path
        })

    # 筛选带✅️的
    selected_probs = [p for p in prob_list if '✅️' in p['title']]

    # 如果带✅️的数量小于50，则继续选择不带❌️且不带✅️的
    if len(selected_probs) < 56:
        additional_probs = [p for p in prob_list if '✅️' not in p['title'] and '❌️' not in p['title']]
        selected_probs.extend(additional_probs[:56 - len(selected_probs)])

    # 创建最终结果字典
    result = {p['title']: {
        "question": p['question'],
        "answer": p['answer'],
        "index": p['index'],
        "NLtype": p['NLtype'],
        "NLChange": p['NLChange'],
        "model_path": p['model_path'],
        "code_path": p['code_path']
    } for p in selected_probs}

    return result


import json


def merge_json_files(file_list, output_filename):
    # 用于存储合并后的内容
    all_data = {}

    # 逐个读取文件内容并合并
    prob_counter = 1  # 用于给新的key按顺序命名，从 prob_001 开始

    for file in file_list:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # 遍历文件中的所有项并重命名key
            for prob_title, prob_data in data.items():
                new_key = f"prob_{str(prob_counter).zfill(3)}"  # 重新命名的key
                all_data[new_key] = prob_data
                prob_counter += 1

    # 将合并后的数据写入新的 JSON 文件
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        json.dump(all_data, outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # 输入的 JSON 文件列表
    file_list = [
        'sample_LP_100_Chinese_A_getAnswer.json',
        'sample_LP_100_Chinese_B_getAnswer.json',
        'sample_LP_100_Chinese_C_getAnswer.json'
    ]
    # 输出的合并后的 JSON 文件
    output_filename = 'ORSample_ABC_200_Chinese.json'

    # 调用函数进行合并
    merge_json_files(file_list, output_filename)

# if __name__ == "__main__":
#     # 输入文件名
#     input_filename = 'sample_LP_100_Chinese_C.md'
#
#     # 调用函数进行转换
#     result = convert_md_to_json(input_filename)
#
#     # 输出到 JSON 文件
#     with open('sample_LP_100_Chinese_C_getAnswer.json', 'w', encoding='utf-8') as json_file:
#         json.dump(result, json_file, ensure_ascii=False, indent=4)
