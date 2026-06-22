import ast
import json
import os
import asyncio
import re

import openai

gemini_model_name = "gemini-3-pro-preview-thinking"

# 假设你已经定义了 BaseAgent，确保传入的 client 是支持异步的 (如 AsyncOpenAI)
class BaseAgent:
    def __init__(self, client, model_name=gemini_model_name, temperature=0.2):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.messages = []
        self.total_tokens = 0

    async def _query(self):
        resp = await self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
        )
        if resp.usage:
            self.total_tokens += resp.usage.total_tokens
        content = resp.choices[0].message.content
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        # self.messages.append({"role": "assistant", "content": content})
        return content


# 裁判 Agent
class EvaluatorAgent(BaseAgent):
    def __init__(self, client, model_name=gemini_model_name, temperature=0.0):
        super().__init__(client, model_name, temperature)
        self.system_prompt = """You are an expert Operations Research reviewer. 
        Your task is to evaluate the extraction results (Parameters, Variables, Constraints, Objective) of an OR problem against the original question and the reference python modeling code.
        Focus on SEMANTIC and LOGICAL correctness, not strict variable name matching.

        Output strictly a JSON object with this exact structure:
        {
            "P": {"total_extracted": 0, "correct": 0},
            "V": {"total_extracted": 0, "correct": 0},
            "C": {"total_extracted": 0, "correct": 0},
            "O": {"total_extracted": 0, "correct": 0},
            "Errors": {
                "Omission": 0,
                "Hallucination": 0,
                "Misalignment": 0
            }
        }

        Definitions:
        - total_extracted: Count of items in the Extracted JSON for that category.
        - correct: Count of extracted items that semantically map to the reference code accurately.
        - Omission: Necessary elements from the code/question that are entirely missing in the extraction.
        - Hallucination: Extracted elements that do not exist or are redundant/meaningless.
        - Misalignment: Extracted elements that are present but mathematically or logically wrong (e.g., wrong inequality direction).
        """

    def _get_fallback_result(self):
        """当大模型彻底发疯，解析完全失败时，返回全0以保证程序继续运行"""
        return {
            "P": {"total_extracted": 0, "correct": 0},
            "V": {"total_extracted": 0, "correct": 0},
            "C": {"total_extracted": 0, "correct": 0},
            "O": {"total_extracted": 0, "correct": 0},
            "Errors": {
                "Omission": 0,
                "Hallucination": 0,
                "Misalignment": 0
            }
        }

    def _robust_json_parse(self, text):
        """极致健壮的 JSON 解析器"""
        if not text:
            return self._get_fallback_result()

        # 尝试 1：直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试 2：清理 Markdown 代码块标记后再解析
        cleaned_text = text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()

        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            pass

        # 尝试 3：使用正则贪婪匹配大括号内部内容
        match = re.search(r'\{[\s\S]*\}', cleaned_text)
        if match:
            json_str = match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # 尝试 4：万不得已，使用 ast 处理单引号或不规范的 Python 字典格式
                try:
                    return ast.literal_eval(json_str)
                except (ValueError, SyntaxError):
                    pass

        # 尝试 5：如果全部失败，打印警告并返回默认值
        print(f"\n[WARNING] JSON Parse Failed. Raw text snippet: {text}...\n")
        return self._get_fallback_result()

    async def evaluate(self, case_id, question, code, extracted_json):
        # 【关键修改】：每次调用 evaluate 时，强制清空历史消息！
        self.messages = []
        # 重新注入干净的 System Prompt
        self.messages.append({"role": "system", "content": self.system_prompt})
        prompt = f"""
Original Question:
{question}

Reference Modeling Code:
{code}

Extracted JSON to Evaluate:
{json.dumps(extracted_json, indent=2)}

Please analyze the extracted JSON and provide the evaluation metrics in the required JSON format, DO NOT output any other things.

**ONLY** Output strictly a JSON object with this exact structure:
{{
    "P": {{"total_extracted": 0, "correct": 0}},
    "V": {{"total_extracted": 0, "correct": 0}},
    "C": {{"total_extracted": 0, "correct": 0}},
    "O": {{"total_extracted": 0, "correct": 0}},
    "Errors": {{
        "Omission": 0,
        "Hallucination": 0,
        "Misalignment": 0
    }}
}}
DO NOT output anything but the JSON object.
"""
        self.messages.append({"role": "user", "content": prompt})

        try:
            res = await self._query()
            return self._robust_json_parse(res)
        except Exception as e:
            # 捕获网络请求错误或其他未预期的异常
            print(f"\n[Case {case_id} ERROR] Model Query Failed: {e}\n")
            return self._get_fallback_result()

async def process_case(case_id, method_name, question, code, extracted_data, client, semaphore):
    """并发执行单个 case 的评估"""
    async with semaphore:  # 控制并发量
        print(f"Starting evaluation: Case {case_id} | Method: {method_name}")
        agent = EvaluatorAgent(client)
        result = await agent.evaluate(case_id, question, code, extracted_data)
        return method_name, result


async def main():
    # 需要替换为你实际的异步 client
    # from openai import AsyncOpenAI
    # client = AsyncOpenAI(api_key="YOUR_API_KEY")

    # API 客户端 Shared async clients
    async_openai = openai.AsyncOpenAI(
        api_key="sk-LF5BgTDbQIGE76923rz4LzodL3PShoOv4HxSYIHsAkNhYtW4",
        base_url="https://api.smai.ai/v1"
    )
    client = async_openai  # 请在此处实例化你的异步客户端
    rootpath = "D:\LLMProject\\NExTORAgent2026\\runs_ALL\【20260219_202525】_【NExTLP】_【gemini-3-flash-preview-high】\\NORA_process_data"
    # 1. 加载数据
    with open(os.path.join(rootpath, 'NExTLP_SbS_o4mini.json'), 'r', encoding='utf-8') as f:
        sbs_data = json.load(f)
    with open(os.path.join(rootpath, 'NExTLP_Direct_Extraction_gemini3flash.json'), 'r', encoding='utf-8') as f:
        direct_data = json.load(f)

    tasks = []
    semaphore = asyncio.Semaphore(40)  # 限制最大并发数为40，防止触发API速率限制

    # 2. 构造并发任务
    for xx_str in sbs_data.keys():
        xx_int = int(xx_str)
        yy = xx_int + 1
        code_path = os.path.join('D:\LLMProject\\NExT2025\\L-problem_code', f'L-prob{yy}.py')

        # 读取参考建模代码
        try:
            with open(code_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
        except FileNotFoundError:
            print(f"Warning: File {code_path} not found. Skipping Case {xx_str}.")
            continue

        question = sbs_data[xx_str].get("question", "")

        # 添加 SbS 任务
        if xx_str in sbs_data:
            tasks.append(process_case(xx_str, "SbS", question, code_content, sbs_data[xx_str], client, semaphore))

        # 添加 Direct 任务
        if xx_str in direct_data:
            tasks.append(process_case(xx_str, "Direct", question, code_content, direct_data[xx_str], client, semaphore))

    # 3. 执行所有任务
    print(f"Executing {len(tasks)} tasks concurrently...")
    results = await asyncio.gather(*tasks)

    # 4. 统计结果
    metrics = {
        "SbS": {"P": [0, 0], "V": [0, 0], "C": [0, 0], "O": [0, 0], "Omission": 0, "Hallucination": 0,
                "Misalignment": 0},
        "Direct": {"P": [0, 0], "V": [0, 0], "C": [0, 0], "O": [0, 0], "Omission": 0, "Hallucination": 0,
                   "Misalignment": 0}
    }

    for method, res in results:
        if not res: continue

        m = metrics[method]
        for key in ["P", "V", "C", "O"]:
            m[key][0] += res.get(key, {}).get("correct", 0)
            m[key][1] += res.get(key, {}).get("total_extracted", 0)

        err = res.get("Errors", {})
        m["Omission"] += err.get("Omission", 0)
        m["Hallucination"] += err.get("Hallucination", 0)
        m["Misalignment"] += err.get("Misalignment", 0)

    # 5. 计算 Precision 并生成 Markdown 表格
    def calc_prec(correct, total):
        return f"{(correct / total) * 100:.1f}%" if total > 0 else "0.0%"

    print("\n\n### 最终统计结果表格 ###\n")
    print(
        "|            | Metric/Precision |               |                 |               | 错误分析  |           |           |")
    print(
        "| ---------- | ---------------- | ------------- | --------------- | ------------- | --------- | --------- | --------- |")
    print(
        "| Method     | Parameters (P)   | Variables (V) | Constraints (C) | Objective (O) | Omission | Hallucination | Misalignment |")

    for method in ["Direct", "SbS"]:
        m = metrics[method]
        p_prec = calc_prec(m["P"][0], m["P"][1])
        v_prec = calc_prec(m["V"][0], m["V"][1])
        c_prec = calc_prec(m["C"][0], m["C"][1])
        o_prec = calc_prec(m["O"][0], m["O"][1])
        print(
            f"| {method.ljust(10)} | {p_prec.ljust(16)} | {v_prec.ljust(13)} | {c_prec.ljust(15)} | {o_prec.ljust(13)} | {str(m['Omission']).ljust(9)} | {str(m['Hallucination']).ljust(13)} | {str(m['Misalignment']).ljust(12)} |")


if __name__ == "__main__":
    asyncio.run(main())