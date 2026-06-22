import json
import random
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Tuple, Literal, Optional
import os

# ---------------- 1.1 环境初始化 -------------------------------------------------
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

openai_api_data = dict(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

# API 客户端
client = AsyncOpenAI(
    api_key=openai_api_data['api_key'],
    base_url=openai_api_data['base_url'] or None
)


# ==========================
# 0. Agent 定义
# ==========================

class BaseAgent:
    def __init__(self, client, model_name="gpt-5.1", temperature=0.2):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.messages = []
        self.total_tokens = 0

    async def _query(self):
        resp = await self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            temperature=self.temperature
        )

        # if resp.usage:
        #     self.total_tokens += resp.usage.total_tokens

        content = resp.choices[0].message.content
        self.messages.append({"role": "assistant", "content": content})
        return content


class Talk_agent(BaseAgent):
    def __init__(self, client, model_name="gpt-5.2", temperature=0.2):
        super().__init__(client, model_name, temperature)
        self.system_msg = (
            '''
           You are an operations optimization expert. 
           Please follow the user's instructions carefully. 
            '''
        )
        self.messages.append({"role": "system", "content": self.system_msg})

    async def generate(self, ask) -> str:
        self.messages.extend([{"role": "user", "content": f"{ask}"}])
        return await self._query()


# ==========================
# 1. 工具函数
# ==========================

def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        print(f"[Warning] File not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


# 假设 Python 版本 >= 3.10 可以使用 str | Path
# 如果 Python 版本 < 3.10，可以使用 Union[str, Path]
def load_file_content(path: str | Path) -> str:
    """
    读取指定路径的文件内容并将其作为字符串返回。
    适用于读取 Python (.py) 或其他文本文件。
    """
    path = Path(path)
    if not path.exists():
        print(f"[Warning] File not found: {path}")
        return ""  # 文件未找到时返回空字符串

    # 使用 f.read() 读取整个文件内容
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"[Error] Failed to read file {path}: {e}")
        return ""

# ==========================
# 2. 数据采样与处理
# ==========================

DATA_CONFIG = {
    "IndustryOR": {
        "summary_path": "data/20251129_ORThought_datasets/summary/summary_industryor.json",
        "processed_root": "data/20251129_ORThought_datasets/processed/IndustryOR",
        "sample_count": 30
    },
    "LogiOR": {
        "summary_path": "data/20251129_ORThought_datasets/summary/summary_logior.json",
        "processed_root": "data/20251129_ORThought_datasets/processed/LogiOR",
        "sample_count": 20
    },
    "NLP4LP": {
        "summary_path": "data/20251129_ORThought_datasets/summary/summary_nlp4lp.json",
        "processed_root": "data/20251129_ORThought_datasets/processed/NLP4LP",
        "sample_count": 50
    }
}


def filter_dataset(data: Dict[str, Any], dataset_name: str) -> List[Dict[str, Any]]:
    """
    过滤逻辑：problem_type != "NLP" AND problem_size in ["Small", "Toy"]
    同时保留原始key以便构建路径
    """
    valid_items = []
    for key, item in data.items():
        p_type = item.get("problem_type", "")
        p_size = item.get("problem_size", "")

        if p_type != "NLP" and p_size in ["Small", "Toy"]:
            # 构造新的结构
            new_item = {
                "description": item.get("description", ""),
                "ground_truth": item.get("ground_truth", 0),
                "problem_type": p_type,
                "problem_size": p_size,
                "dataset_name": dataset_name,
                "original_key": key  # 保存原始 key (如 prob_001) 用于寻找路径
            }
            valid_items.append(new_item)
    return valid_items


def sample_lp_100_process(out_path: str | Path, seed: int = 42) -> None:
    random.seed(seed)
    combined_items = []

    # 1. 按照顺序读取并采样
    for ds_name, config in DATA_CONFIG.items():
        raw_data = load_json(config["summary_path"])
        filtered = filter_dataset(raw_data, ds_name)

        count = config["sample_count"]
        if len(filtered) < count:
            print(f"[Warning] {ds_name} has only {len(filtered)} valid items, requesting {count}. Taking all.")
            sampled = filtered
        else:
            sampled = random.sample(filtered, count)

        # 2. 格式化并添加路径
        for item in sampled:
            orig_key = item["original_key"]  # e.g., "prob_000"

            # 构建路径
            # 假设结构: processed_root / prob_000 / model.txt
            proc_root = Path(config["processed_root"])
            model_p = proc_root / orig_key / "model.txt"
            code_p = proc_root / orig_key / "code.py"

            final_item = {
                "description": item["description"],
                "ground_truth": item["ground_truth"],
                "problem_type": item["problem_type"],
                "problem_size": item["problem_size"],
                "index": f"{ds_name}_{orig_key}",  # IndustryOR_prob_001
                "model_path": str(model_p),
                "code_path": str(code_p)
            }
            combined_items.append(final_item)

    # 3. 保存结果，刷新 key 为 prob_001 ... prob_100
    out_data = {}
    for idx, item in enumerate(combined_items, 1):
        new_key = f"prob_{idx:03d}"
        out_data[new_key] = item

    save_json(out_data, out_path)
    print(f"[sample_lp_100] Saved {len(out_data)} items to {out_path}")


# ==========================
# 3. 翻译模块
# ==========================

async def translate_to_target(client, text: str, target_lang: str = "Chinese") -> str:
    agent = Talk_agent(client)
    instruction = (
        f"Please translate the following Operations Research problem description into {target_lang} precisely. "
        "Keep all constraints, numbers, and objectives unchanged. "
        "Do NOT solve it. Output ONLY the translated text."
    )
    ask = {
        "translate_task": instruction,
        "target_language": target_lang,
        "content": text
    }
    # 为了简化，直接把 ask 转字符串给 agent，或者你可以修改 agent 接收结构化 prompt
    return (await agent.generate(ask)).strip()


async def translate_CtoE_target(client, text: str, target_lang: str = "Chinese") -> str:
    agent = Talk_agent(client)
    instruction = (
        f"Please translate the following Operations Research problem description into {target_lang} precisely. "
        "Keep all constraints, numbers, and objectives unchanged. "
        "注意，遇到公式需要使用$...$包裹，"
        "此外，必须按照数学领域、优化领域的方向进行翻译，不能简单进行翻译成英语，例如专业词汇`幂`和`次方`等。"
        "Do NOT solve it. Output ONLY the translated text. "
    )
    ask = {
        "translate_task": instruction,
        "target_language": target_lang,
        "content": text
    }
    # 为了简化，直接把 ask 转字符串给 agent，或者你可以修改 agent 接收结构化 prompt
    return (await agent.generate(ask)).strip()

async def build_translated_dataset(client, input_path: str, out_path: str, target_lang="Chinese") -> None:
    data = load_json(input_path)
    out = {}

    # 排序保证顺序处理
    keys = sorted(data.keys(), key=lambda x: int(x.split('_')[-1]))

    # 并发控制，防止速率限制
    sem = asyncio.Semaphore(10)

    async def process_item(k, item):
        async with sem:
            # 根据方向选择字段
            src_text = item.get("description") if target_lang == "Chinese" else item.get("question")

            # 如果是非线性数据集翻译回英文，可能字段叫 question
            if not src_text and "question" in item:
                src_text = item["question"]


            if target_lang == 'English':
                trans = await translate_CtoE_target(client, src_text, target_lang)
            else:
                trans = await translate_to_target(client, src_text, target_lang)

            new_item = dict(item)
            # 统一把翻译后的结果放到 'question' 字段（如果是中文）
            # 或者覆盖 'question' 字段（如果是回译英文）
            new_item["question"] = trans
            if target_lang == "Chinese":
                # 保留 original description 用于参考? 这里按需求只需存入
                pass
            return k, new_item

    pending = []
    for k in keys:
        pending.append(process_item(k, data[k]))

    results = await asyncio.gather(*pending)

    for k, res_item in results:
        out[k] = res_item

    save_json(out, out_path)
    print(f"[Translation] Saved to {out_path} ({target_lang})")


# ==========================
# 4. 非线性注入 (A/B/C) 以及代码的初始修改
# ==========================

def get_deterministic_operator_instruction():
    # 1. 随机生成 1 到 10 的整数，
    choice = random.randint(1, 10)
    if choice in [1, 2, 3, 4, 5]:
        # 非整数幂方
        result_str = (
            "A: 【引入非线性算子：非整数幂方/高次方】\n"
            "请选择一个线性成本或约束项，引入非整数幂方算子。\n"
            "具体操作：实际应用中可能需要使用一个修正系数/修正函数幂，例如将 '3x' 改为 k=1.2 的幂形式，即 '(3x)^1.2'。只改一处！"
        )
        op_type = "非整数幂方"

    elif choice in [6, 7]:
        # 高次方变量/多变量相乘
        result_str = (
            "A: 【引入非线性算子：高次方/多变量耦合】\n"
            "请选择一个目标函数项，将其从总数增改为多个（>=3个）决策变量相乘。\n"
            "具体操作：在现有数学模型里，从改动某个约束或者目标函数，将简单的 $y=kx$ 变为 $y=k \\cdot x \\cdot z \\cdot t$ 等等，其中 k 为系数，x, z, t 都为决策变量。"
        )
        op_type = "高次方 (立方/乘积)"

    elif choice in [8, 9]:
        # 指对数
        result_str = (
            "A: 【引入非线性算子：指数/对数】\n"
            "请选择一个目标约束条件，将其改为满足指数增长。\n"
            "具体操作：设定某个变量其实存在增长系数（例如 1.3或者0.9），使该项呈现指数级变化特征。"
        )
        op_type = "指对数"

    else:  # choice == 4
        # 三角函数
        result_str = (
            "A: 【引入非线性算子：三角函数 (sin/cos)】\n"
            "请选择一个目标约束条件，使其在一定范围内具有周期性。\n"
            "具体操作：在原有基础上，某个决策变量的数值按照正弦函数进行浮动。"
            "例如让模型的约束从 $y=5x$ 变成 $y=5x(0.1 * cos(πx/n * t) + 1),即x会随着周期、月或者季度等时间t以及自身变量x共同影响下微微变动，n为常数,"
            "若没有提及周期、月或者季度等时间概念，则需要添加一些时间概念。$。"
        )
        op_type = "三角函数"

    return choice, op_type, result_str

async def inject_nonlinearity_logic(client, question_zh: str, nl_type: str) -> Tuple[str, str]:
    agent = Talk_agent(client)

    _, _, final_string = get_deterministic_operator_instruction()

    instructions = {
        "A": final_string,
        "B": "【引入分式/比率】将某一个线性总量约束改为使用分式的比率描述或平均值约束。"
             "具体操作：例如'单位A小于等于100倍的单位B'改为'某几个变量和某几个变量的比率小于100'。或者x<=10y 转换描述为--> x/y<=10。只改一处!",
        "C": "【引入逻辑/指示符】引入一个或多个条件判断。"
             "具体操作：例如'如果生产量超过X，则额外产生固定成本Y或边际成本1000'。"
    }

    prompt = (
        f"Task: Inject specific non-linearity (Type {nl_type}) into the original question in Chinese Language.\n"
        f"Rule: {instructions.get(nl_type)}\n"
        "Requirements:\n"
        "1. KEEP the rest of the problem unchanged!!!\n"
        "2. Use natural language descriptions to describe the injected non-linearity. NOT use complex LaTeX formulas if possible. \n"
        "3. 第一个返回值是'full_modified_question'。 即完整的引入非线性修改后的题目完整的表述。"
        "其中，描述必须使用具体的数值，即修改的部分若引入新的参数或常数，则必须要给出具体的数值，例如当题目出现给定系数/常数k，则之后必须还要声明k的值，假设k=.. \n"
        "4. 第二个返回值是'part_of_changed_description'。即必须标明修改了哪里，怎么改的，格式固定为：'原题目相关描述:...., 修改后相关描述:....'"
        "5. ONLY Output format JSON: {\"full_modified_question\": \"...\", \"part_of_changed_description\": \"...\"}"
    )

    resp = await agent.generate({"instruction": prompt, "original_question": question_zh})

    try:
        # 清洗一下 markdown 格式
        clean_resp = resp.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_resp)
        return parsed["full_modified_question"], parsed["part_of_changed_description"]
    except:
        return question_zh, f"Failed to parse LLM response, LLM resp={resp}"

async def generate_modified_code(client, desc:str, NL_desc:str, original_code: str, nl_change: str, nl_type: str) -> str:
    """
    让 LLM 尝试修改某个代码 (Step 2.1)
    """
    agent = Talk_agent(client)
    prompt = (
        f"Original description: {desc}\n"
        f"Original Gurobi Python code is provided.\n"
        f"original_code: {original_code}"
        f"A non-linear change (Type {nl_type}) was made to the problem: {nl_change}\n"
        f"And the FULL Non-linear changed description is provided.\n"
        f"Non-linear changed description: {NL_desc}\n"
        f"Attention:\n"
        "1. Specifically, first, comment out the part of code in the original code that needs modification,"
        " write a comment on the line above it, with the fixed format: # ❤ Non-linearity is introduced. ❤ "
        "Then, add the code that introduces non-linearity.\n"
        "3. Output the complete modified Python code (including the import section, the mathematical model function section, and the `if __name__ == '__main__'` section), instead of just outputting the code for the non-linear modification.\n"
        "4. Output Needs this format block: ```python\n ....(full modified Python code) \n```"
    )
    return await agent.generate(prompt)

# --- 新编写的函数：保存修改后的代码 ---
def save_modified_code(code_content: str, nl_type: str, idx: int):
    """
    将生成的 Python 代码 (code_content) 保存到指定的路径结构中。
    路径结构：data/processed/sample_LP_100_Chinese_{nl_type}/prob_{idx:03d}.py

    Args:
        code_content: 要保存的 Python 代码字符串。
        nl_type: 非线性类型 (e.g., "A", "B", "C")。
        idx: 当前处理项的索引 (用于文件名)。
    """

    # 构造文件夹路径：data/processed/sample_LP_100_Chinese_{nl_type}
    # 严格按照用户指定的路径结构：data/processed/sample_LP_100_Chinese_{nl_type}
    nl_folder_name = f"sample_LP_100_Chinese_{nl_type}"
    base_nonlinear_path = Path("data/processed") / nl_folder_name

    # 确保目标文件夹（包括父文件夹）存在
    base_nonlinear_path.mkdir(parents=True, exist_ok=True)

    # 构造最终文件名：prob_{idx:03d}.py
    file_name = f"prob_{idx:03d}.py"
    final_path = base_nonlinear_path / file_name

    # 写入文件
    try:
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(code_content)
        print(f"    [Code Saved] {final_path}")
    except Exception as e:
        print(f"[Error] Failed to save code to {final_path}: {e}")


# --- 原有的主函数 (已修改集成新功能) ---
async def build_nonlinear_datasets(client, input_path: str, output_base_path: str, sem_num: int = 20) -> None:
    data = load_json(input_path)
    types = ["A", "B", "C"]

    # 1. 设置并发信号量，防止瞬间请求过多导致 429 错误
    # 根据你的 API 限制，建议设置为 10 ~ 50 之间
    sem = asyncio.Semaphore(sem_num)

    async def process_single_item(idx: int, original_key: str, item: dict, nl_type: str):
        """处理单个条目的内部函数，用于并发调用"""
        async with sem:
            q_zh = item.get("question", "")
            asw = item.get("ground_truth", "")
            original_code = load_file_content(item["code_path"])
            # 调用耗时的 LLM 逻辑
            mod_q, desc = await inject_nonlinearity_logic(client, q_zh, nl_type)
            mod_code = await generate_modified_code(client, desc, mod_q, original_code, desc, nl_type)
            mod_code = mod_code.replace("```python", "").replace("```", "").strip()
            # ************************************************
            # 2. 调用新函数：保存生成的代码 (mod_code)
            # ************************************************
            save_modified_code(mod_code, nl_type, idx)

            # 3. 组装结果数据
            new_key = f"prob_{idx:03d}"
            new_item = {
                "question": mod_q,
                "answer": asw,
                "index": item["index"],
                "NLtype": nl_type,
                "NLChange": desc
            }
            # 保留引用路径
            if "model_path" in item: new_item["model_path"] = item["model_path"]

            # 注意：这里需要更新 code_path 指向新的代码文件路径
            nl_folder_name = f"sample_LP_100_Chinese_{nl_type}"
            new_item["code_path"] = str(Path("data/processed") / nl_folder_name / f"prob_{idx:03d}.py")

            return new_key, new_item

    # 2. 外层按类型循环（A -> B -> C）
    # 如果想更极致，这里也可以把 ABC 放到一起并发，但分开处理逻辑更清晰，且单类型的并发已足够跑满带宽
    for nl_type in types:
        keys = sorted(data.keys())
        tasks = []

        print(f"[NonLinear-{nl_type}] Generating tasks for {len(keys)} items...")

        # 3. 创建并发任务列表
        for idx, k in enumerate(keys, 1):
            tasks.append(process_single_item(idx, k, data[k], nl_type))

        # 4. 并行执行所有任务
        # results 会按照 tasks 的顺序返回结果，即 [(key1, item1), (key2, item2), ...]
        results = await asyncio.gather(*tasks)

        # 5. 组装数据并保存
        out_data = dict(results)

        final_path = f"{output_base_path}_{nl_type}.json"
        save_json(out_data, final_path)
        print(f"[NonLinear-{nl_type}] Saved metadata to {final_path}")

# ==========================
# 5. Golden Code & Answer (人工介入模拟)
# ==========================

def expert_handle(dataset_path: str, out_path: str) -> None:
    """
    Step 2.2: 模拟人工 + NORA 修正代码并运行得到答案的过程。
    真实场景下：这里会暂停，等待你手动修改文件，或者调用一个能够执行代码的 Agent 循环修正。
    这里我们做一个‘虚拟’操作：
    1. 读取 json
    2. 假设 code 已经修好了 (mock)
    3. 假设运行后得到了新的 answer (mock)
    """
    data = load_json(dataset_path)
    print(f"\n--- [EXPERT HANDLE START] Processing {dataset_path} ---")
    print("User Action Required: Review codes, fix errors, run Gurobi, get answers.")

    # 模拟人工操作：更新 answer
    # 在实际流程中，你会手动编辑这个 json 或者用另一个脚本跑完更新它
    for k, item in data.items():
        # 模拟：简单地把 'wait_to_get' 替换成一个随机数，或者标记为 'HANDLED'
        # 实际代码中，你应该在这里读取你存好的正确答案
        item["answer"] = "42.0"  # Mock Answer
        # item["golden_code"] = "..." # 如果需要存代码

    save_json(data, out_path)
    print(f"--- [EXPERT HANDLE END] Simulated answers saved to {out_path} ---\n")


# ==========================
# 6. 整合与实验
# ==========================

def merge_datasets(linear_path, a_path, b_path, c_path, out_path):
    # 加载所有 JSON 文件
    d_lin = load_json(linear_path)
    d_a = load_json(a_path)
    d_b = load_json(b_path)
    d_c = load_json(c_path)

    all_items = []

    def collect(d, tag):
        # 按 prob_001 ... 排序
        keys = sorted(d.keys(), key=lambda x: int(x.split('_')[-1]))
        for k in keys:
            item = d[k]

            # 确保每个 item 至少有这些字段
            if "NLtype" not in item:
                item["NLtype"] = "Linear"

            # 删除不需要的字段
            item.pop("problem_type", None)
            item.pop("problem_size", None)

            # 将 description 改为 question
            if "description" in item:
                item["question"] = item.pop("description")

            # 将 ground_truth 改为 answer
            if "ground_truth" in item:
                item["answer"] = item.pop("ground_truth")

            # 确保所有必要的字段存在，如果缺少某个字段则填充默认值
            required_fields = ["question", "answer", "index", "model_path", "code_path"]
            for field in required_fields:
                if field not in item:
                    item[field] = None

            all_items.append(item)

    # 收集每个数据源的信息
    collect(d_lin, "Linear")
    collect(d_a, "A")
    collect(d_b, "B")
    collect(d_c, "C")

    final_data = {}
    # 创建新的 prob_xxx 键并将所有项存入最终字典
    for i, item in enumerate(all_items, 1):
        new_key = f"prob_{i:03d}"
        final_data[new_key] = item

    # 保存合并后的数据到目标路径
    save_json(final_data, out_path)

    print(f"[Merge] Total {len(final_data)} items saved to {out_path}")


async def run_experiment_logic(dataset_path):
    data = load_json(dataset_path)

    # 模拟结果容器
    results = {
        "Linear": {"AC": 0, "PR": 0, "errors": {"semantic": 0, "nonlinear": 0, "code": 0}},
        "A": {"AC": 0, "PR": 0, "errors": {"semantic": 0, "nonlinear": 0, "code": 0}},
        "B": {"AC": 0, "PR": 0, "errors": {"semantic": 0, "nonlinear": 0, "code": 0}},
        "C": {"AC": 0, "PR": 0, "errors": {"semantic": 0, "nonlinear": 0, "code": 0}},
    }

    print("\n=== Running Experiments (Simulated) ===")
    # 遍历 400 题
    for k, item in data.items():
        ntype = item.get("NLtype", "Linear")

        # 1. 这里的逻辑是调用 DeepSeek/OptiMUS/LLMOPT
        # pred_code = await call_model(item["question"])
        # status, val = execute_gurobi(pred_code)

        # 2. 模拟结果
        is_correct = random.random() > 0.3  # 假设 70% 正确率
        is_runnable = True

        if is_correct:
            results[ntype]["AC"] += 1
            results[ntype]["PR"] += 1
        else:
            # 随机分配错误类型
            err_type = random.choice(["semantic", "nonlinear", "code"])
            results[ntype]["errors"][err_type] += 1
            if err_type != "code":
                results[ntype]["PR"] += 1  # 只有 code error 算 PR 失败

    # 归一化
    for k in results:
        count = 100  # 每一类100题
        results[k]["AC"] /= count
        results[k]["PR"] /= count

    print(json.dumps(results, indent=2))
    print("TODO: Generate Plots based on this JSON.")


# ==========================
# 主流程 Pipeline
# ==========================

async def main():
    # 路径定义
    path_lp_100 = "data/processed/sample_LP_100.json"
    # path_lp_100_ch = "data/processed/sample_LP_100_Chinese.json"

    # # 1. 采样 100 条
    # print("--- Step 1: Sampling ---")
    # sample_lp_100_process(path_lp_100)
    #
    # # 2. 翻译成中文 (Linear)
    # print("--- Step 2: Translating to Chinese ---")
    # await build_translated_dataset(client, path_lp_100, path_lp_100_ch, target_lang="Chinese")

    # # 3. 注入非线性 (A, B, C) 并尝试修改代码
    # print("--- Step 3: Injecting Nonlinearity ---")
    # base_nonlinear_path = "data/processed/sample_LP_100_Chinese"
    # await build_nonlinear_datasets(client, path_lp_100_ch, base_nonlinear_path)
    #
    # print("--- End ---")
    #
    # 4. Golden Code & Answer (人工环节)
    # print("--- Step 4: Expert Handle (Golden Answer) ---")

    paths_ch_abc = {
        "A": "data/processed/sample_LP_100_Chinese_A.json",
        "B": "data/processed/sample_LP_100_Chinese_B.json",
        "C": "data/processed/sample_LP_100_Chinese_C.json"
    }
    paths_ans_abc = {
        "A": "data/processed/sample_LP_100_Chinese_A_getAnswer.json",
        "B": "data/processed/sample_LP_100_Chinese_B_getAnswer.json",
        "C": "data/processed/sample_LP_100_Chinese_C_getAnswer.json"
    }

    # for t in ["A", "B", "C"]:
    #     # 这一步包含了 2.1 (自动改代码) 和 2.2 (人工修正) 的逻辑封装
    #     expert_handle(paths_ch_abc[t], paths_ans_abc[t])

    # 5. 翻译回英文
    print("--- Step 5: Translating back to English ---")
    paths_en_abc = {}
    for t in ["A", "B", "C"]:
        out_p = paths_ans_abc[t] + "_to_En.json"
        await build_translated_dataset(client, paths_ans_abc[t], out_p, target_lang="English")
        paths_en_abc[t] = out_p

    # 6. 整合最终数据集
    print("--- Step 6: Merging to ORSample_LABC_300 ---")
    final_path = "data/processed/ORSample_LABC_300.json"
    merge_datasets(
        path_lp_100,  # 原版线性 (英文)
        paths_en_abc["A"],
        paths_en_abc["B"],
        paths_en_abc["C"],
        final_path
    )
    #
    # # 7. 实验
    # print("--- Step 7: Experiment ---")
    # await run_experiment_logic(final_path)


if __name__ == "__main__":
    # 确保文件夹存在
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    asyncio.run(main())