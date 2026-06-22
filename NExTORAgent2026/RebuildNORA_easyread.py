# -*- coding: utf-8 -*-
"""
Async version of or_llm_eval_multi_agents.py with agent classes for modeling, coding, and repair using openai.AsyncOpenAI.
"""
import datetime
import os
import json
import argparse
import asyncio
from tqdm import tqdm
import re
import time
import numpy as np
from collections import defaultdict

# ===== Unified entry & helpers =====
from typing import Optional
from enum import Enum


import openai
from dotenv import load_dotenv

from RebuildNORA_utils import (
    is_number_string,
    async_extract_and_execute_python_code,
    eval_model_result
)
from agents.base_agents import Simple_agent

from agents.model_agents import ModelingAgent,AuxiliaryModelAgent
from agents.coding_agents import CodingAgent
from agents.repair_agents import RepairAgent
from agents.extract_info_agents import ExtractFactorsFromLongTextAgent, ExtractParameterFromLongTextAgent, \
    DirectExtractALL

# ---------------- 1.1 环境初始化 -------------------------------------------------
load_dotenv()
# ---------------- 1.2 API 客户端初始化 -------------------------------------------
openai_api_data = dict(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)
# API 客户端 Shared async clients
async_openai = openai.AsyncOpenAI(
    api_key=openai_api_data['api_key'],
    base_url=openai_api_data['base_url'] or None
)

async def call_openai(system_prompt: str, prompt: str, model=None):
    model = model if model is not None else "gpt-5"
    resp = await async_openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    answer = resp.choices[0].message.content.strip()
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
    return answer

async def call_openai_for_judge_error_type(question, code, error_msg):
    """
    让 LLM 判断错误属于哪一类，并强制返回 int 类型的 1, 2, or 3。
    """
    system_prompt = (
        "你是一个资深运筹学专家。你的任务是分类错误类型。"
        "请只输出一个数字：1、2 或 3。"
        "不要输出任何标点符号、解释或额外的文字。"
    )

    prompt = f"""
    I am analyzing why an optimization code failed.
    Problem: {question}
    Code: {code}
    Error Message: {error_msg}

    Classify the error into one of these 3 types:
    Type 1: Modeling Semantic Error (Code can PASS,but Logic is wrong, or Infeasible model).
    Type 2: Non-linear Constraint Error (e.g., "Objective must be linear or quadratic", x*x, x/y in linear solver).
    Type 3: Other Coding Error (Syntax error, missing variables, API misuse).

    Output Requirement:
    - Return ONLY the number (1, 2, or 3).
    - Do not write "Type 1", just "1".
    """

    # 调用大模型
    response_content = await call_openai(system_prompt, prompt)

    # --- 核心改进部分：清洗与提取 ---

    # 1. 确保是字符串
    text = str(response_content)

    # 2. 使用正则表达式寻找文本中出现的第一个 1, 2 或 3
    # r"\b([1-3])\b" 意思是在单词边界匹配 1-3，防止匹配到 10, 123 等
    # 如果模型可能输出 "Type1" (没有空格)，则使用 r"([1-3])" 更宽容
    match = re.search(r"([1-3])", text)

    if match:
        result = int(match.group(1))
        return result

    # 3. 兜底逻辑 (Fallback)
    # 如果模型完全疯了，输出了一堆无关的话，或者没找到数字
    # 根据你的源代码逻辑，默认返回 3 (Other Coding Error) 是比较安全的
    print(f"Warning: LLM output '{text}' could not be parsed. Defaulting to 3.")
    return 3






# Code Solver:生成与修复循环
async def async_code_solver(coder, repairer, entry, math_model,
                            max_attempts=3, side_info=None, opts=None):
    runs_path = opts.output_dir
    coding = coder
    repair = repairer
    gurobi_code = await coding.generate(entry, math_model)
    print("【Python Gurobi Code】:\n", gurobi_code)
    attempt = 0
    while attempt < max_attempts:
        print(f"\n第 {attempt + 1} 次尝试，开始执行代码...\n")
        success, result = await async_extract_and_execute_python_code(gurobi_code, entry, runs_path, attempt)
        if success:
            return True, result, gurobi_code
        # 修复专家关闭
        # else:
        #     return False, None, gurobi_code
        # 修复专家启动
        print(f"\n第 {attempt + 1} 次尝试失败，请求 LLM 修复代码...\n")
        advise = await repair.generate(entry, math_model, result, gurobi_code)
        new_gurobi_code = await coding.generate(entry, math_model, analysis=advise, gurobi_code=gurobi_code,
                                                side_info=side_info)
        gurobi_code = new_gurobi_code
        print("\n获取到修复后的代码，准备重新执行...\n")
        attempt += 1

    return False, None, gurobi_code



# ----------------------------------------------------------------------------------------------------------------------
# ---------------- 4. 主要流程 ------------------------- #################################################################
# ----------------------------------------------------------------------------------------------------------------------

async def async_NExT_OR_Agent(entry, opts, max_attempts=3):
    model_name = opts.model
    # 1) 先调用建模专家
    modeler = ModelingAgent(async_openai, model_name=model_name)
    math_model = await modeler.generate(entry)
    print("【Mathematical Model】", math_model)
    problem_type = entry.get('Problem_Type', '')

    # 2) 调用代码建议与修复专家
    if problem_type == "NLP":
        print("NLP problem,use【Auxiliary Mathematical Agent】")
        auxiliary_modeler = AuxiliaryModelAgent(async_openai, model_name=model_name)
        math_model = await auxiliary_modeler.integrate_model(entry, math_model)
    else:
        auxiliary_modeler = None

    # 3) 调用写代码以及修复专家，生成并调试 Gurobi 代码
    coder = CodingAgent(async_openai, model_name=model_name, problem_type=problem_type)
    repair = RepairAgent(async_openai, model_name=model_name)

    # 内部函数，用于在函数返回前计算总tokens
    def get_total_tokens():
        tokens = modeler.total_tokens
        if auxiliary_modeler:
            tokens += auxiliary_modeler.total_tokens
        tokens += coder.total_tokens
        tokens += repair.total_tokens
        return tokens

    # 4) 若成功或重试逻辑 unchanged...
    success, result, _ = await async_code_solver(coder, repair, entry, math_model,
                                                 max_attempts=max_attempts, side_info=None, opts=opts)
    print(f'Stage result: {success}, {result}')
    if result is None:
        result = [None]
    if success:
        if is_number_string(str(result[0])):
            # 得到最优解
            return True, result, get_total_tokens()
        else:
            # 没有最优解
            print('!![Run no available solution warning]!!')
            side_info = (
                "The model code still reports errors after multiple debugging attempts. Please carefully check if "
                "there are errors in the mathematical model. After checking, please rebuild the Gurobi Python code. "
                "Output in the format \n```python\n{code}\n```, without code explanations."
            )
            success, result, _ = await async_code_solver(coder, repair, entry, math_model,
                                                         max_attempts=3, side_info=side_info,
                                                         opts=opts)
            return success, result, get_total_tokens()
    else:
        # Run no success
        print('!![Run no success]!!')
        side_info = (
            "The model code still reports errors after multiple debugging attempts. Please carefully check if "
            "there are errors in the mathematical model. After checking, please rebuild the Gurobi Python code. "
            "Output in the format \n```python\n{code}\n```, without code explanations."
        )
        success, result, _ = await async_code_solver(coder, repair, entry, math_model,
                                                     max_attempts=3, side_info=side_info,
                                                     opts=opts)
    return success, result, get_total_tokens()



async def async_gpt_code_agent_simple(entry, opts, max_attempts=3):
    model_name = opts.model
    runs_path = opts.output_dir
    """
    Async version of gpt_code_agent_simple
    """
    simple_agent = Simple_agent(async_openai, model_name=model_name)
    gurobi_code = await simple_agent.generate(entry)

    print("【Python Gurobi 代码】:\n", gurobi_code)
    text = f"{gurobi_code}"
    is_solve_success, result = await async_extract_and_execute_python_code(text, entry)

    print(f'Stage result: {is_solve_success}, {result}')

    # 内部函数，用于在函数返回前计算总tokens
    def get_total_tokens():
        tokens = simple_agent.total_tokens
        return tokens

    return is_solve_success, result, get_total_tokens(), gurobi_code

# ----------------- 主流程与并发执行 -----------------------------------------
async def process_single_case(i, entry, args):
    start_time = time.perf_counter()
    print(f"=== Case {i} ===")
    q, ans = entry['question'], entry['answer']
    print(q)
    print('-------------')

    # 统计错误类型
    error_types = {1: 0, 2: 0, 3: 0}  # 统计不同错误类型的计数

    output_dir = args.output_dir
    guro_code = None
    error_msg = None
    # 初始化结果字典
    result_data = {
        "entry": entry,
        "execution": {},
        "evaluation": {}
    }
    res = 0
    ok = True
    total_tokens = 0
    # 执行代码
    if args.agent:
        ok, res, total_tokens = await async_NExT_OR_Agent(entry, args)
    else:
        ok, res, total_tokens, guro_code = await async_gpt_code_agent_simple(entry, args)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    if isinstance(res, list):
        for res_i in res:
            if len(str(res_i)) > 20:
                error_msg = res
                res = None
                break
    elif isinstance(res, str):
        if len(res) > 20:
            error_msg = res
            res = None
    print("res=", res)

    # 记录执行结果
    if ok:
        print(f"成功执行代码，最优解值: {res}")
        result_data["execution"]["status"] = "success"
        result_data["execution"]["result"] = res
    else:
        print("执行代码失败。")

        result_data["execution"]["status"] = "failed"
        result_data["execution"]["result"] = res if 'res' in locals() else None

    result_data["execution"]["time_seconds"] = elapsed_time
    result_data["execution"]["total_tokens"] = total_tokens

    pass_flag, correct_flag = eval_model_result(ok, res, ans)

    result_data["evaluation"] = {
        "pass_flag": pass_flag,
        "correct_flag": correct_flag,
        "run_result": res,
        "ground_truth": ans

    }
    if pass_flag:
        if not correct_flag:
            error_types[1] += 1
    else:
        # 调用 GPT 判断错误类型

        # error_type = await call_openai_for_judge_error_type(
        #     "分析代码运行失败的原因", guro_code, error_msg
        # )
        error_type = 1
        # 更新错误类型的计数
        error_types[int(error_type)] += 1

    print(f"Result: solve={ok}, value={res}, ground_truth={ans}, time={elapsed_time:.2f}s, tokens={total_tokens}")
    print(f'[Final] {i}run pass: {pass_flag}, solve correct: {correct_flag}')
    print(' ')

    # 保存结果到文件
    filename = f"case_{entry['index']}.txt"
    temp_file_path = os.path.join(output_dir, filename)
    with open(temp_file_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    return pass_flag, correct_flag, i, elapsed_time, total_tokens, error_types



async def process_data_case(i, entry, args):
    print(f"============ Case {i} For Data Process ============")
    para_agent = ExtractParameterFromLongTextAgent(async_openai, model_name=args.model)
    var_agent = ExtractFactorsFromLongTextAgent(async_openai, model_name=args.model)

    para_entry = await para_agent.integrate_with_file(entry)
    full_entry = await var_agent.integrate_with_file(para_entry)
    print(f"============ 【END】 Case {i} For Data Process ============")
    return full_entry, i


async def process_direct_data_case(i, entry, args):
    print(f"============ Case {i} For Data Process ============")
    direct_agent = DirectExtractALL(async_openai, model_name=args.model)
    question = entry.get('question', '')
    direct_result = await direct_agent.generate(question)
    print(f"============ 【END】 Case {i} For Data Process ============")
    return direct_result, i


async def main_all():
    opts = await get_args()

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    runs_ALL_path = os.path.join("runs_ALL", f"【{timestamp}】_【{opts.dataset_name}】_【{opts.model}】")
    os.makedirs(runs_ALL_path, exist_ok=True)
    os.makedirs(runs_ALL_path + "//NORA_process_data", exist_ok=True)
    opts.output_dir = runs_ALL_path

    base, ext = os.path.splitext(opts.dataset_name)
    opts.nora_file_input_path = f"{opts.output_dir}//NORA_process_data//{base}_NORA.json"

    await main_process_data(opts)
    await main(opts)


async def main_process_data(opts=None):
    if opts is None:
        opts = await get_args()

    if opts.agent:
        opts.data_path = os.path.join("data/20251021_origin_datasets", f"{opts.dataset_name}" + ".json")

    base, ext = os.path.splitext(opts.dataset_name)
    new_v_ext = opts.model if opts.model else "o4mini_simple"
    # 构建新的路径
    if opts.output_dir is None:
        nora_path = os.path.join("NORA_process_data", f"{os.path.basename(base)}_NORA_{new_v_ext}.json")
    else:
        nora_path = opts.nora_file_input_path

    input_path = opts.data_path
    print(f"input_path={input_path}, nora_path={nora_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # tasks = [process_data_case(i, entry, opts) for i, entry in dataset.items()]
    tasks = []
    for i, d in dataset.items():
        # task = process_data_case(i, d, opts)
        task = process_direct_data_case(i, d, opts)
        tasks.append(task)

    total = len(tasks)
    # ↓↓↓↓↓ 这里创建一个 tqdm 实例 ↓↓↓↓↓
    # position=0 → 把进度条固定在最顶行（position=1 就是第二行，依此类推）
    # leave=True  → 任务完成后保留进度条
    pbar = tqdm(
        total=total,
        desc="总进度",
        position=0,
        leave=True,
        ncols=100,  # 可选：固定宽度
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )
    fails = {}
    # 并发等待每个子任务完成
    for coro in asyncio.as_completed(tasks):
        full_entry, i = await coro

        fails[str(i)] = full_entry
        # 推动进度条，并更新右侧的 postfix
        pbar.update(1)
        # 如果你还要打印每个 case 的细节，可以用 tqdm.write() 保证它们
        # 打印在进度条上方，而不会破坏进度条本身的位置：
        # tqdm.write(f"Case {idx:>3}: pass={p}, correct={c}")
    pbar.close()

    with open(nora_path, 'w', encoding='utf-8') as f:
        json.dump(fails, f, ensure_ascii=False, indent=4)
    print(f"Integrated ALL data into {nora_path}")


async def main(opts=None):
    if opts is None:
        opts = await get_args()

    # opts.output_dir = "runs_ALL\【20250627_011339】_【optmath_bench_LP】_【o4-mini-2025-04-16-high】"
    # opts.nora_file_input_path = "runs_ALL/【20250627_011339】_【optmath_bench_LP】_【o4-mini-2025-04-16-high】/NORA_process_data/optmath_bench_LP_NORA_o4mini_simple.json"

    if opts.nora_file_input_path is None:
        if opts.agent:
            new_v_ext = "o4mini"
            opts.data_path = "NORA_process_data/" + opts.dataset_name + f"_NORA_{new_v_ext}.json"
        else:
            opts.data_path = os.path.join("data/processed", f"{opts.dataset_name}" + ".json")
    else:
        opts.data_path = opts.nora_file_input_path

    input_path = opts.data_path
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if opts.output_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        opts.output_dir = os.path.join("runs", f"【{timestamp}】_【{opts.dataset_name}】_【{opts.model}】")
        os.makedirs(opts.output_dir, exist_ok=True)

    tasks = []
    for i, entry in dataset.items():
        # # w/o 逐句参数匹配机制
        # entry["Parameters_List"]=[]
        # # w/o 长文本要素匹配机制
        # entry["Sentence_Scanning"] = []
        # entry["Variables_List"] = []
        # entry["Constraint_Table"] = []
        # # entry["Objective"] = []
        # # w/o 非线性识别机制 & 辅助变量模型
        # entry["Problem_Type"] = "MILP"
        task = process_single_case(i, entry, opts)
        tasks.append(task)

    total = len(tasks)
    pass_count = 0
    correct_count = 0

    # ↓↓↓↓↓ 这里创建一个 tqdm 实例 ↓↓↓↓↓
    # position=0 → 把进度条固定在最顶行（position=1 就是第二行，依此类推）
    # leave=True  → 任务完成后保留进度条
    pbar = tqdm(
        total=total,
        desc="总进度",
        position=0,
        leave=True,
        ncols=100,  # 可选：固定宽度
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )
    fails = []
    error_types_list = []
    # 并发等待每个子任务完成
    for coro in asyncio.as_completed(tasks):
        p, c, idx,_elapsed_time, _total_tokens, error_types = await coro

        # 更新计数
        if p:
            pass_count += 1
        if c:
            correct_count += 1
        if not (p and c):
            fails.append(idx)
        if error_types:
            error_types_list.append(error_types)
        # 推动进度条，并更新右侧的 postfix
        pbar.update(1)
        pbar.set_postfix({
            "pass": pass_count,
            "correct": correct_count
        }, refresh=True)

        # 如果你还要打印每个 case 的细节，可以用 tqdm.write() 保证它们
        # 打印在进度条上方，而不会破坏进度条本身的位置：
        # tqdm.write(f"Case {idx:>3}: pass={p}, correct={c}")
    pbar.close()


    # ==================== 2026.1.5 BEGIN ACL新加错误类型统计（三种错误）
    def calculate_error_type_ratios(error_types_list) -> None:
        """
        计算并打印 1、2、3 类错误的比重。

        参数:
        results: 包含所有文件处理结果的字典列表，每个字典包含错误类型统计信息
        """
        error_type_counts = {1: 0, 2: 0, 3: 0}  # 初始化错误类型计数器

        # 遍历所有文件处理结果，累计每种错误类型的数量
        for e_t in error_types_list:
            for error_type, count in e_t.items():
                error_type_counts[error_type] += count

        total_errors = sum(error_type_counts.values())  # 计算总错误数
        if total_errors > 0:
            # 打印每种错误类型的比重
            for error_type, count in error_type_counts.items():
                print(f"错误类型 {error_type} 的比重: {count / total_errors * 100:.2f}%")
        else:
            print("没有发现任何错误。")

    calculate_error_type_ratios(error_types_list)
    print("error_types_list:", error_types_list)
    # ==================== 2026.1.5 END ACL新加错误类型统计（三种错误）


    print(f"[Total {len(dataset)}] pass: {pass_count}, correct: {correct_count}")
    print(f"Failed cases: {fails}")
    AAA_result_path = os.path.join(opts.output_dir, f"AAA_result_{len(dataset)}_{pass_count}_{correct_count}.txt")
    AAA_result = {
        f"[Total {len(dataset)}] pass": pass_count,
        f"[Total {len(dataset)}] correct": correct_count,
        "Failed cases": fails
    }
    with open(AAA_result_path, 'w', encoding='utf-8') as f:
        json.dump(AAA_result, f, ensure_ascii=False, indent=4)


async def main_round_10(opts=None, num_rounds: int = 10):
    """
    将你原先的 main_round_10 泛化：
    - 移除内部对 dataset_name / nora 路径的硬编码
    - 允许通过参数控制轮数
    其余逻辑保持不变
    """
    if opts is None:
        opts = await get_args()

    accuracies = []
    instance_times = defaultdict(list)
    instance_tokens = defaultdict(list)

    # == 复用 main 的数据定位规则 ==
    if getattr(opts, "nora_file_input_path", None) is None:
        if opts.agent:
            new_v_ext = "o4mini"
            opts.data_path = os.path.join("NORA_process_data", f"{opts.dataset_name}_NORA_{new_v_ext}.json")
        else:
            opts.data_path = os.path.join("data/NExT_datasets", f"{opts.dataset_name}.json")
    else:
        opts.data_path = opts.nora_file_input_path

    with open(opts.data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    base_run_dir = "runs_10_rounds"
    os.makedirs(base_run_dir, exist_ok=True)

    for round_num in range(num_rounds):
        print(f"\n{'=' * 20} ROUND {round_num + 1}/{num_rounds} {'=' * 20}\n")
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        round_output_dir = os.path.join(base_run_dir, f"round_{round_num + 1}_{timestamp}")
        os.makedirs(round_output_dir, exist_ok=True)
        opts.output_dir = round_output_dir

        tasks = [process_single_case(i, entry, opts) for i, entry in dataset.items()]
        correct_count = 0
        total_count = len(tasks)

        pbar = tqdm(total=total_count, desc=f"Round {round_num + 1}", ncols=100)
        for coro in asyncio.as_completed(tasks):
            # 统一为 5 元组解包（与你的 process_single_case 返回保持一致）
            pass_flag, correct_flag, i, elapsed_time, total_tokens = await coro
            if correct_flag:
                correct_count += 1
            instance_times[str(i)].append(elapsed_time)
            instance_tokens[str(i)].append(total_tokens)
            pbar.update(1)
        pbar.close()

        round_accuracy = correct_count / total_count if total_count > 0 else 0
        accuracies.append(round_accuracy)
        print(f"\n--- Round {round_num + 1} Summary ---")
        print(f"Accuracy: {round_accuracy:.4f}")

    # --- 10轮结束后，计算并打印最终统计数据 ---
    print(f"\n{'=' * 20} FINAL STATISTICS (after {num_rounds} rounds) {'=' * 20}\n")

    # 准确率与方差
    avg_accuracy = np.mean(accuracies)
    var_accuracy = np.var(accuracies)
    print(f"Overall Accuracy: {avg_accuracy:.2%}")
    print(f"Accuracy Variance: {var_accuracy:.6f}")

    # 每个实例的平均用时和token
    print("\n--- Per-Instance Average Statistics ---")
    avg_instance_times = {i: np.mean(times) for i, times in instance_times.items()}
    avg_instance_tokens = {i: np.mean(tokens) for i, tokens in instance_tokens.items()}

    # --- 过滤步骤：只保留满足条件的实例 ---
    filtered_times = {i: t for i, t in avg_instance_times.items() if t < 300}
    filtered_tokens = {i: avg_instance_tokens[i] for i in filtered_times.keys() if avg_instance_tokens[i] <= 32 * 1024}

    # 同步过滤（只保留两个条件都满足的实例）
    valid_ids = set(filtered_times.keys()) & set(filtered_tokens.keys())
    filtered_times = {i: avg_instance_times[i] for i in valid_ids}
    filtered_tokens = {i: avg_instance_tokens[i] for i in valid_ids}

    # 计算过滤后总体平均
    overall_avg_time = np.mean(list(filtered_times.values())) if filtered_times else 0
    overall_avg_tokens = np.mean(list(filtered_tokens.values())) if filtered_tokens else 0

    print("\n--- Filtered Statistics (tokens<=32K, time<300s) ---")
    print(f"Valid Instances: {len(valid_ids)} / {len(avg_instance_times)}")
    print(f"Filtered Avg Time: {overall_avg_time:.2f} seconds")
    print(f"Filtered Avg Tokens: {overall_avg_tokens:.0f}")


    # 将最终统计结果保存到文件
    final_stats = {
        "overall_accuracy_mean": avg_accuracy,
        "accuracy_variance": var_accuracy,
        "average_instance_times": avg_instance_times,
        "average_instance_tokens": avg_instance_tokens,
        "round_accuracies": accuracies,
        "filtered_avg_time": overall_avg_time,
        "filtered_avg_tokens": overall_avg_tokens,
        "filtered_instance_count": len(valid_ids),

    }
    stats_filename = os.path.join(base_run_dir, f"final_stats_{opts.dataset_name}_{opts.model}.json")
    with open(stats_filename, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=4)
    print(f"\nFinal statistics saved to {stats_filename}")


async def single_main_index(index: int, opts=None):
    """
    单个 index 的最小化调度器：不解析 CLI，仅按 opts 里的参数运行。
    """
    if opts is None:
        opts = await get_args()  # 复用你的现成参数对象

    # == 与 main 的输入判定逻辑保持一致 ==
    # 若显式传入 nora 文件则直接用；否则按 agent 分支构造默认路径
    if getattr(opts, "nora_file_input_path", None) is None:
        if opts.agent:
            new_v_ext = "gpt-5"
            opts.data_path = os.path.join("NORA_process_data", f"{opts.dataset_name}_NORA_{new_v_ext}.json")
        else:
            opts.data_path = os.path.join("data/NExT_datasets", f"{opts.dataset_name}.json")
    else:
        opts.data_path = opts.nora_file_input_path

    # 仅加载与 index 对应的条目
    with open(opts.data_path, "r", encoding="utf-8") as f:
        dataset_all = json.load(f)
    # 兼容 str/int 索引
    key = str(index)
    if key not in dataset_all:
        raise KeyError(f"index {index} 不在数据集中（键不存在）")
    dataset = {key: dataset_all[key]}

    # 输出目录
    if getattr(opts, "output_dir", None) is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        opts.output_dir = os.path.join("runs_single", f"【{timestamp}】_【{opts.dataset_name}】_【{opts.model}】_【{index}】")
    os.makedirs(opts.output_dir, exist_ok=True)

    # 调度
    tasks = [process_single_case(key, dataset[key], opts)]
    results = await asyncio.gather(*tasks)
    pass_count = sum(1 for p, _, _, _, _ in results if p)
    correct_count = sum(1 for _, c, _, _, _ in results if c)
    print(f"[Single index={index}] pass={pass_count}, correct={correct_count}")

    return results



class RunMode(str, Enum):
    PROCESS = "process"   # 仅抽取：等价 main_process_data
    RUN     = "run"       # 仅求解：等价 main
    ROUNDS  = "rounds"    # 多轮：   等价 main_round_10（可指定轮数）
    SINGLE  = "single"    # 单例：   等价 single_main（参数化 index）
    ALL     = "all"       # 端到端： 等价 main_all


async def run_unified(opts):
    """
    统一入口：根据 mode 调用不同主流程。
    """
    # process / all 模式：如果没给 output_dir，就按 main_all 的规则建 runs_ALL 目录
    if opts.mode in (RunMode.PROCESS, RunMode.ALL, "process", "all"):
        if opts.output_dir is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            runs_ALL_path = os.path.join("runs_ALL", f"【{timestamp}】_【{opts.dataset_name}】_【{opts.model}】")
            os.makedirs(runs_ALL_path, exist_ok=True)
            os.makedirs(runs_ALL_path + "//NORA_process_data", exist_ok=True)
            opts.output_dir = runs_ALL_path
        # 若未显式指定 nora 输出文件名，则按 main_process_data 的默认命名生成
        if opts.nora_file_input_path is None:
            base, _ = os.path.splitext(opts.dataset_name)
            new_v_ext = opts.model
            opts.nora_file_input_path = f"{opts.output_dir}//NORA_process_data//{base}_NORA_{new_v_ext}.json"

    # 分发
    m = RunMode(opts.mode)
    if m is RunMode.PROCESS:
        await main_process_data(opts)  # 仅抽取
    elif m is RunMode.RUN:
        await main(opts)               # 仅求解
    elif m is RunMode.ROUNDS:
        await main_round_10(opts=opts, num_rounds=opts.rounds)  # 多轮
    elif m is RunMode.SINGLE:
        if opts.index is None:
            raise ValueError("SINGLE 模式需要提供 index")
        await single_main_index(index=opts.index, opts=opts)
    elif m is RunMode.ALL:
        await main_process_data(opts)  # 端到端
        await main(opts)
    else:
        raise ValueError(f"未知 mode: {opts.mode}")



def get_args():
    args = argparse.ArgumentParser(description='Async OR LLM multi-agent solver')
    args.add_argument('--agent', action='store_true',
                      help='use multi-agent repair loop')
    args.add_argument('--model', type=str, default='gpt-5.1')
    args.add_argument('--dataset_name', type=str, default='NExTLP',
                      help='Name of the dataset to be processed')
    opts = args.parse_args()
    opts.output_dir = None
    opts.nora_file_input_path = None # "NORA_process_data/NExT_NLP_NORA_o4mini.json"
    opts.model = "gemini-3-flash-preview-high"
    opts.mode = RunMode.PROCESS
    opts.index = None
    # opts.agent = False
    return opts

if __name__ == "__main__":
    '''
    # 1) 仅抽取
    await run_unified("process", dataset_name="optmath_bench_LP", model="o4-mini", agent=True)
    # 2) 仅求解（读取已有 NORA）
    await run_unified("run", dataset_name="optmath_bench_LP", nora_file_input_path="NORA_process_data/optmath_bench_LP_NORA_o4mini.json")
    # 3) 多轮评测（10 轮）
    await run_unified("rounds", dataset_name="nl4opt_LP", rounds=10)
    # 4) 单例 index 调用
    await run_unified("single", dataset_name="NExT_LP", index=16)
    # 5) 端到端
    await run_unified("all", dataset_name="optmath_bench_LP", model="o4-mini", agent=True)
    '''

    opts = get_args()
    asyncio.run(run_unified(opts))
