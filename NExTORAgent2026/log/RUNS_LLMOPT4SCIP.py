import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path
import re
import os
from NEXT_utils import eval_model_result

from typing import List, Optional, Tuple, Dict, Any

five_description_simple = "The five-element model is the abstraction of an optimization problem, which transforms specific problem scenarios into formal mathematical problems. You need to write the corresponding Pyomo code based on the five-element model provided. "

five_description_code = "The following is the five-element model of an optimization problem: "

generate_system_info = "You are an expert in the field of operations and optimization. You need to complete some optimization problem modeling tasks."


def F2CS(five: str, index: int) -> str:
    text = five_description_simple + five_description_code + five + f"""
Please write the corresponding SCIP code.
Please add from pyscipopt import Model at the beginning of your code (you may add other imports as needed).

When naming, please use model for model naming, that is, you must use: model = Model("xxxx").
Please do NOT solve the model, do NOT write model.optimize().
Please do not output the running log.
SCIP 不能使用矩阵间的乘法或者加减法, 必须使用 for 循环进行逐项添加约束。此外, 连续性变量的变量类型应该写成 C 而不是 CONS, 例如: vtype='C'.

You need to export the LP file at the end of the script using: model.writeProblem("case_{index}_output.lp").

You must write the code in the form of a class and include a main function.
在类中, 变量和参数都必须在初始化的时候定义, 可以在类的全局中使用.
"""
    return text

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
path = "/home/user/Desktop/equastar/deployment/models/ss_model"
path_t = "/home/user/Desktop/equastar/deployment/models/ss_model"
device = "cuda"

generator = pipeline(
    "text-generation",
    path,
    torch_dtype="auto",
    device_map="auto",
)

print("Model Load Success.")


def extract_result(stdout: str) -> Optional[float]:
    if not stdout:
        return None

    text = stdout.strip()

    pattern = r"objective value\s+([+-]?\d+\.\d+e[+-]?\d+|\d+\.\d+|\d+)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)

    if not matches:
        return None

    vals: List[float] = []
    for m in matches:
        try:
            vals.append(float(m))
        except Exception:
            pass

    if not vals:
        return None

    clean_vals: List[float] = []
    for v in vals:
        scaled = v / 1e5
        clean_vals.append(scaled)

    result = min(clean_vals, key=lambda x: abs(x))
    return result




def infer_code(five_elem: str, index: int) -> str:
    messages = [
        {
            "role": "user",
            "content": F2CS(five_elem, index),
        }
    ]
    outputs = generator(messages, max_new_tokens=32768)[0]["generated_text"]
    response = outputs[-1]["content"]

    with open("model-ss-case.txt", "w", encoding="utf-8") as file:
        file.write(str(response))

    text = response if isinstance(response, str) else str(response)

    code_blocks = re.findall(
        "[\\x60]{3}(?:python)?\\s*(.*?)[\\x60]{3}",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if code_blocks:
        code = code_blocks[-1].strip()
    else:
        code = text.strip()

    return code


import asyncio
from asyncio.subprocess import PIPE

CASE_BOUND_PATTERN = re.compile(
    r"(case_(\\d+)_output\\.lp).*?(?:dual bound|primary bound)\\s*:\\s*([+-]?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)",
    re.IGNORECASE | re.DOTALL,
)


def parse_bounds_from_text(text: str) -> Dict[int, float]:
    results: Dict[int, float] = {}
    for match in CASE_BOUND_PATTERN.finditer(text):
        idx = int(match.group(2))
        val = float(match.group(3))
        results[idx] = val
    return results


async def _generate_one_code_file(
    entry: Dict[str, Any],
    code_dir: Path,
    semaphore: asyncio.Semaphore,
) -> Tuple[int, str]:
    idx = entry["index"]
    question = entry.get("question", "")

    async with semaphore:
        five_elem = await asyncio.to_thread(infer_five_elem, question)
        code_str = await asyncio.to_thread(infer_code, five_elem, idx)

        save_path = code_dir / f"case_{idx}.py"
        await asyncio.to_thread(save_path.write_text, code_str, encoding="utf-8")

        print(f"[code] case_{idx} -> {save_path}")
        return idx, str(save_path)


async def generate_code_files_async(
    dataset: Dict[str, Dict[str, Any]],
    code_dir: Path,
    max_concurrency: int = 4,
) -> Dict[int, str]:
    code_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = [
        _generate_one_code_file(entry, code_dir, semaphore)
        for _, entry in dataset.items()
    ]

    results = await asyncio.gather(*tasks)
    mapping: Dict[int, str] = {idx: path for idx, path in results}
    print(f"[code] generated {len(mapping)} scripts in {code_dir}")
    return mapping


async def _run_single_py_to_lp(
    py_file: Path,
    semaphore: asyncio.Semaphore,
) -> Tuple[Path, int, Optional[Path]]:
    async with semaphore:
        print(f"[lp] run {py_file}")
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(py_file),
            cwd=str(py_file.parent),
            stdout=PIPE,
            stderr=PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            print(f"[lp][ERROR] {py_file.name} exit {proc.returncode}")
            if stderr:
                print(stderr.decode(errors="ignore"))

        lp_path: Optional[Path] = None
        match = re.search(r"case_(\\d+)\\.py", py_file.name)
        if match:
            idx = match.group(1)
            candidate = py_file.parent / f"case_{idx}_output.lp"
            if candidate.exists():
                lp_path = candidate

        if lp_path is None:
            lp_files = sorted(
                py_file.parent.glob("*.lp"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if lp_files:
                lp_path = lp_files[0]

        if lp_path:
            print(f"[lp] {py_file.name} -> {lp_path.name}")
        else:
            print(f"[lp][WARN] no lp found in {py_file.parent}")

        return py_file, proc.returncode, lp_path


async def generate_lp_files_async(
    code_dir: Path,
    max_concurrency: int = 4,
) -> Dict[str, Optional[str]]:
    if not code_dir.exists():
        raise FileNotFoundError(f"code_dir not found: {code_dir}")

    py_files = sorted(code_dir.glob("*.py"))
    if not py_files:
        print(f"[lp] no .py files in {code_dir}")
        return {}

    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        _run_single_py_to_lp(py_file, semaphore)
        for py_file in py_files
    ]

    results = await asyncio.gather(*tasks)
    mapping: Dict[str, Optional[str]] = {}
    for py_file, rc, lp_path in results:
        mapping[py_file.name] = str(lp_path) if lp_path else None

    print(f"[lp] processed {len(mapping)} .py files")
    return mapping


async def run_solver_shell_async(
    work_dir: Path,
    timeout: int = 3600,
) -> Dict[int, float]:
    if not work_dir.exists():
        raise FileNotFoundError(f"work_dir not found: {work_dir}")

    cmd = ["bash", "run", "test_ai_solver.sh"]
    print(f"[run] cwd={work_dir}, cmd={' '.join(cmd)}")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(work_dir),
        stdout=PIPE,
        stderr=PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"Command {' '.join(cmd)} timeout({timeout}s)")

    stdout_text = stdout.decode(errors="ignore") if isinstance(stdout, bytes) else stdout
    stderr_text = stderr.decode(errors="ignore") if isinstance(stderr, bytes) else stderr

    if proc.returncode != 0:
        print(f"[run][WARN] exit code: {proc.returncode}")
        if stderr_text:
            print("[run][STDERR]:")
            print(stderr_text)

    results = parse_bounds_from_text(stdout_text)
    if not results:
        results = parse_bounds_from_text(stderr_text)

    print(f"[run] parsed {len(results)} case results")
    for idx, val in sorted(results.items()):
        print(f"case_{idx}_output.lp -> {val}")

    return results


def NEXT_main():
    parser = argparse.ArgumentParser(description="Async OR LLM multi-stage runner")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["code", "lp", "run"],
        help="code: generate code; lp: run py to generate lp; run: call external solver script",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="NExT_LP",
        help="used in mode=code for path naming",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="json path for mode=code; if None, will be constructed from dataset_name",
    )
    parser.add_argument(
        "--code_dir",
        type=str,
        default=None,
        help="mode=code: dir to save py; mode=lp: dir containing py files",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="mode=run: work dir to run shell script",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="max concurrency",
    )

    opts = parser.parse_args()

    if opts.mode == "code":
        if opts.dataset_path is None:
            input_root = "D:\\LLMProject\\LLMOPT-main\\data\\NEXTOR"
            opts.dataset_path = os.path.join(input_root, f"{opts.dataset_name}.json")

        with open(opts.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if opts.code_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            root = Path("MY_inference") / "runs" / f"【{timestamp}】_【{opts.dataset_name}】_LLMOPT"
            code_dir = root / "codes"
        else:
            code_dir = Path(opts.code_dir)

        asyncio.run(
            generate_code_files_async(
                dataset,
                code_dir,
                max_concurrency=opts.max_workers,
            )
        )

    elif opts.mode == "lp":
        if opts.code_dir is None:
            raise ValueError("mode=lp requires --code_dir")
        code_dir = Path(opts.code_dir)
        asyncio.run(
            generate_lp_files_async(
                code_dir,
                max_concurrency=opts.max_workers,
            )
        )

    elif opts.mode == "run":
        work_dir = Path(opts.work_dir or ".")
        asyncio.run(run_solver_shell_async(work_dir))


if __name__ == "__main__":
    NEXT_main()
