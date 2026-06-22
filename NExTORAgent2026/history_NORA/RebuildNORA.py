# -*- coding: utf-8 -*-
"""
Async version of or_llm_eval_multi_agents.py with agent classes for modeling, coding, and repair using openai.AsyncOpenAI.
"""
import datetime
import os
import sys
import json
import argparse
import tempfile
import asyncio
from tqdm import tqdm
import re
import time
import numpy as np
from collections import defaultdict



import openai
from dotenv import load_dotenv

from utils import (
    is_number_string,
    async_extract_and_execute_python_code,
    eval_model_result
)

from agents.model_agents import ModelingAgent,AuxiliaryModelAgent
from agents.coding_agents import CodingAgent
from agents.repair_agents import RepairAgent
from agents.extract_info_agents import ExtractFactorsFromLongTextAgent,ExtractParameterFromLongTextAgent

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


########################################################################################################################
########################################################################################################################
########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# ---------------- 4. 主要流程 ------------------------- #################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################
########################################################################################################################
########################################################################################################################

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


# ----------------- 主流程与并发执行 -----------------------------------------
async def process_single_case(i, entry, args):
    start_time = time.perf_counter()
    print(f"=== Case {i} ===")
    q, ans = entry['question'], entry['answer']
    print(q)
    print('-------------')

    output_dir = args.output_dir

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

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    if isinstance(res, list):
        for res_i in res:
            if len(str(res_i)) > 20:
                res = None
                break
    elif isinstance(res, str):
        if len(res) > 20:
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

    print(f"Result: solve={ok}, value={res}, ground_truth={ans}, time={elapsed_time:.2f}s, tokens={total_tokens}")
    print(f'[Final] {i}run pass: {pass_flag}, solve correct: {correct_flag}')
    print(' ')

    # 保存结果到文件
    filename = f"case_{entry['index']}.txt"
    temp_file_path = os.path.join(output_dir, filename)
    with open(temp_file_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    return pass_flag, correct_flag, i, elapsed_time, total_tokens



async def process_data_case(i, entry, args):
    print(f"============ Case {i} For Data Process ============")
    var_agent = ExtractFactorsFromLongTextAgent(async_openai, model_name=args.model)
    para_agent = ExtractParameterFromLongTextAgent(async_openai, model_name=args.model)

    para_entry = await para_agent.integrate_with_file(entry)
    full_entry = await var_agent.integrate_with_file(para_entry)
    print(f"============ 【END】 Case {i} For Data Process ============")
    return full_entry, i


async def get_args():
    args = argparse.ArgumentParser(description='Async OR LLM multi-agent solver')
    args.add_argument('--agent', action='store_true',
                      help='use multi-agent repair loop')
    args.add_argument('--model', type=str, default='o4-mini')
    args.add_argument('--dataset_name', type=str, default='optmath_bench_LP',
                      help='Name of the dataset to be processed')
    opts = args.parse_args()
    opts.multipara = False
    opts.output_dir = None
    opts.nora_file_input_path = None
    return opts


async def main_all():
    opts = await get_args()
    opts.multipara = True

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    runs_ALL_path = os.path.join("runs_ALL", f"【{timestamp}】_【{opts.dataset_name}】_【{opts.model}】")
    os.makedirs(runs_ALL_path, exist_ok=True)
    os.makedirs(runs_ALL_path + "//NORA_process_data", exist_ok=True)
    opts.output_dir = runs_ALL_path

    base, ext = os.path.splitext(opts.dataset_name)
    new_v_ext = "o4mini_simple"
    opts.nora_file_input_path = f"{opts.output_dir}//NORA_process_data//{base}_NORA_{new_v_ext}.json"

    await main_process_data(opts)
    await main(opts)


async def main_process_data(opts=None):
    if opts is None:
        opts = await get_args()

    if opts.agent:
        opts.data_path = os.path.join("data/origin_data", f"{opts.dataset_name}" + ".json")

    base, ext = os.path.splitext(opts.dataset_name)
    new_v_ext = "o4mini_simple"
    # 构建新的路径
    if opts.output_dir is None:
        nora_path = os.path.join("../NORA_process_data", f"{os.path.basename(base)}_NORA_{new_v_ext}.json")
    else:
        nora_path = opts.nora_file_input_path

    input_path = opts.data_path
    print(f"input_path={input_path}, nora_path={nora_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # tasks = [process_data_case(i, entry, opts) for i, entry in dataset.items()]
    tasks = []
    for i, d in dataset.items():
        task = process_data_case(i, d, opts)
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


async def _wrapper(task_coro, progress):
    res = await task_coro
    progress.update(1)
    return res


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
            opts.data_path = os.path.join("data/NExT_datasets", f"{opts.dataset_name}" + ".json")
    else:
        opts.data_path = opts.nora_file_input_path

    input_path = opts.data_path
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if opts.output_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        opts.output_dir = os.path.join("../runs", f"【{timestamp}】_【{opts.dataset_name}】_【{opts.model}】")
        os.makedirs(opts.output_dir, exist_ok=True)

    tasks = []
    for i, entry in dataset.items():
        # w/o 逐句参数匹配机制
        # entry["Parameters_List"]=[]
        # w/o 长文本要素匹配机制
        # entry["Sentence_Scanning"] = []
        # entry["Variables_List"] = []
        # entry["Constraint_Table"] = []
        # entry["Objective"] = []
        # w/o 非线性识别机制 & 辅助变量模型
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
    # 并发等待每个子任务完成
    for coro in asyncio.as_completed(tasks):
        p, c, idx = await coro

        # 更新计数
        if p:
            pass_count += 1
        if c:
            correct_count += 1
        if not (p and c):
            fails.append(idx)
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


async def main_round_10(opts=None):
    if opts is None:
        opts = await get_args()

    # 用于统计多轮结果的数据结构
    accuracies = []
    instance_times = defaultdict(list)
    instance_tokens = defaultdict(list)

    opts.dataset_name = "nl4opt_LP"
    opts.nora_file_input_path = "../NORA_process_data/nl4opt_LP_NORA.json"
    num_rounds = 2

    # 假设数据已由 main_process_data 处理好，直接加载
    if opts.nora_file_input_path is None:
        if opts.agent:
            new_v_ext = "o4mini"
            opts.data_path = "NORA_process_data/" + opts.dataset_name + f"_NORA_{new_v_ext}.json"
        else:
            opts.data_path = os.path.join("data/NExT_datasets", f"{opts.dataset_name}" + ".json")
    else:
        opts.data_path = opts.nora_file_input_path

    input_path = opts.data_path
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)


    base_run_dir = "runs_10_rounds"
    os.makedirs(base_run_dir, exist_ok=True)

    for round_num in range(num_rounds):
        print(f"\n{'=' * 20} ROUND {round_num + 1}/{num_rounds} {'=' * 20}\n")

        # 为每一轮创建一个独立的输出目录
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        round_output_dir = os.path.join(base_run_dir, f"round_{round_num + 1}_{timestamp}")
        os.makedirs(round_output_dir, exist_ok=True)
        opts.output_dir = round_output_dir

        tasks = []
        for i, entry in dataset.items():
            # # 此处逻辑与 main() 保持一致
            # entry["Sentence_Scanning"] = []
            # entry["Variables_List"] = []
            # entry["Constraint_Table"] = []
            # entry["Objective"] = []
            task = process_single_case(i, entry, opts)
            tasks.append(task)

        correct_count = 0
        total_count = len(tasks)

        # 并发执行当前轮次的所有任务
        pbar = tqdm(total=total_count, desc=f"Round {round_num + 1}", ncols=100)
        for coro in asyncio.as_completed(tasks):
            pass_flag, correct_flag, i, elapsed_time, total_tokens = await coro
            if correct_flag:
                correct_count += 1

            # 存储每个实例的用时和token
            instance_times[str(i)].append(elapsed_time)
            instance_tokens[str(i)].append(total_tokens)
            pbar.update(1)
        pbar.close()

        # 计算并记录当前轮次的准确率
        round_accuracy = correct_count / total_count if total_count > 0 else 0
        accuracies.append(round_accuracy)
        print(f"\n--- Round {round_num + 1} Summary ---")
        print(f"Accuracy: {round_accuracy:.2%}")
        print(f"---------------------------\n")

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


async def single_main():
    args = argparse.ArgumentParser(description='Async OR LLM multi-agent solver')
    args.add_argument('--agent', action='store_true',
                      help='use multi-agent repair loop')
    args.add_argument('--model', type=str, default='o4-mini')
    args.add_argument('--dataset_name', type=str, default='NExT_LP',
                      help='Name of the dataset to be processed')
    args.add_argument('--singe_test', type=str, default="16",
                      help='Name of the dataset to be processed')
    args.add_argument('--test_nora', type=str, default=None,
                      help='Name of the dataset to be processed')
    opts = args.parse_args()
    opts.test_nora = "【20250619_155846】_【NExT_LP】_【o4-mini-2025-04-16-high】_【16】"
    # opts.test_nora = None
    # 构建新的路径
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    opts.output_dir = os.path.join("../runs_single",
                                   f"【{timestamp}】_【{opts.dataset_name}】_【{opts.model}】_【{opts.singe_test}】")
    os.makedirs(opts.output_dir, exist_ok=True)
    nora_path = os.path.join(opts.output_dir, f"{opts.dataset_name}_{str(opts.singe_test)}_NORA.json")

    task_list = []
    if opts.singe_test is not None and opts.singe_test != "":
        task_list = [str(x.strip()) for x in opts.singe_test.split(",")]
    print("task_list=", task_list)

    if opts.test_nora is not None:
        print("opts.test_nora is not None, and use exist file.")
        exist_path = os.path.join("../runs_single", opts.test_nora,
                                  f"{opts.dataset_name}_{str(opts.singe_test)}_NORA.json")
        with open(exist_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    else:
        print("opts.test_nora is None, and create a new file.")
        input_path = os.path.join("data/origin_data", f"{opts.dataset_name}" + ".json")
        with open(input_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        single_tasks = []
        for i, d in dataset.items():
            if len(task_list) >= 1 and str(i) in task_list:
                task = process_data_case(i, d, opts)
                single_tasks.append(task)
        results = await asyncio.gather(*single_tasks)

        fails = {str(i): full_entry for full_entry, i in results if full_entry}

        print(f"input_path={input_path}, nora_path={nora_path}")

        with open(nora_path, 'w', encoding='utf-8') as f:
            json.dump(fails, f, ensure_ascii=False, indent=4)
        print(f"Integrated ALL data into {nora_path}")
        dataset = fails

    single_tasks_runs = []
    for i, entry in dataset.items():
        if len(task_list) >= 1 and str(i) in task_list:
            task = process_single_case(i, entry, opts)
            single_tasks_runs.append(task)
    results = await asyncio.gather(*single_tasks_runs)

    pass_count = sum(1 for p, _, _ in results if p)
    correct_count = sum(1 for _, c, _ in results if c)
    fails = [i for p, c, i in results if not (p and c)]
    print(f"[Total {len(dataset)}] pass: {pass_count}, correct: {correct_count}")
    print(f"Failed cases: {fails}")


if __name__ == "__main__":
    asyncio.run(main_process_data())
    # asyncio.run(main_all())

