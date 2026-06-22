import json
import sys
import asyncio
import tempfile
import os
from typing import Optional, Union, Any
import re
import math
from typing import List, Tuple

from openai import AsyncOpenAI

DEFAULT_SUBPROCESS_TIMEOUT = 200

_Number = Union[int, float]

def convert_to_number(s: object) -> Optional[_Number]:
    """Best-effort conversion. Trims spaces, accepts +/- and scientific notation."""
    if s is None:
        return None
    if not isinstance(s, str):
        # allow direct numbers to pass through
        if isinstance(s, (int, float)) and not isinstance(s, bool):
            return int(s) if isinstance(s, float) and s.is_integer() else s
        s = str(s)
    s = s.strip()
    try:
        # try float first; reduce branchiness
        v = float(s)
        return int(v) if v.is_integer() else v
    except ValueError:
        return None

def is_number_string(s: object) -> bool:
    return convert_to_number(s) is not None


# -------------------------------抓取答案:extract_best_objective------------------------------------------
import re
import math
from typing import List, Tuple

# 预编译正则
_RE_FINAL = re.compile(
    r'FinalAnswer=【\s*([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)\s*】'
)
_RE_OPTIMAL = re.compile(
    r'Optimal objective\s+([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)',
    re.IGNORECASE
)
_RE_BEST = re.compile(
    r'Best objective\s+([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)',
    re.IGNORECASE
)

def _safe_float(s: str) -> float | None:
    """安全地将字符串转为 float，过滤 NaN/Inf。"""
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def _dedup_with_tolerance(values_in_order: List[float],
                          rel_tol: float,
                          abs_tol: float) -> List[float]:
    """按出现顺序做容差去重"""
    dedup: List[float] = []
    for v in values_in_order:
        if not dedup:
            dedup.append(v)
            continue
        if any(math.isclose(v, u, rel_tol=rel_tol, abs_tol=abs_tol) for u in dedup):
            continue
        dedup.append(v)
    return dedup

def extract_best_objective(output_text: str,
                           *,
                           include_best: bool = True,
                           rel_tol: float = 1e-9,
                           abs_tol: float = 1e-9) -> List[str]:
    """
    从输出文本中提取所有可能的目标值，返回列表（若没抓到则返回 ["None"]）。

    捕获内容：
    - FinalAnswer=【...】
    - Optimal objective ...
    - (可选) Best objective ...

    参数：
    - include_best: 是否提取 Best objective
    - rel_tol / abs_tol: 去重时的容差
    """
    if not output_text:
        return ["None"]

    candidates: List[Tuple[int, float]] = []

    # FinalAnswer
    for m in _RE_FINAL.finditer(output_text):
        v = _safe_float(m.group(1))
        if v is not None:
            candidates.append((m.start(1), v))

    # Optimal objective
    for m in _RE_OPTIMAL.finditer(output_text):
        v = _safe_float(m.group(1))
        if v is not None:
            candidates.append((m.start(1), v))

    # Best objective (optional)
    if include_best:
        for m in _RE_BEST.finditer(output_text):
            v = _safe_float(m.group(1))
            if v is not None:
                candidates.append((m.start(1), v))

    # 若未找到，返回 ["None"]
    if not candidates:
        return ["None"]

    # 保持出现顺序
    candidates.sort(key=lambda kv: kv[0])
    values_in_order = [v for _, v in candidates]

    # 容差去重
    unique_vals = _dedup_with_tolerance(values_in_order, rel_tol=rel_tol, abs_tol=abs_tol)

    # 转成字符串返回（保证类型统一）
    return [f"{v}" for v in unique_vals]



#  -------------------eval_model_result:判断答案ground_truth和result是否一致---------------------

import math
from typing import Any, List, Tuple, Union, Optional

NumberLike = Union[int, float, str]

def _to_list(x: Any) -> List[Any]:
    """统一输入为列表格式。"""
    return x if isinstance(x, list) else [x]

def _is_no_solution(x: Any) -> bool:
    """
    判断输入是否表示“无解/不可行”。
    支持：'infeasible', 'no solution', 'none', 'nan', 'inf', 'infinity'
    """
    if x is None:
        return True
    if isinstance(x, (int, float)):
        if math.isnan(x) or math.isinf(x):
            return True
        return False
    s = str(x).strip().lower()
    return s in {"none", "infeasible", "no solution", "nan", "inf", "infinity"}

def _to_float(x: Any) -> Optional[float]:
    """
    尽力转换为 float；
    - 若为“无解”字符串则返回 None；
    - 若为非法/非有限值也返回 None。
    """
    if _is_no_solution(x):
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        v = float(x)
        return v if math.isfinite(v) else None
    try:
        v = float(str(x).strip())
        return v if math.isfinite(v) else None
    except Exception:
        return None

def _mantissa_base10(v: float) -> Tuple[float, int]:
    """返回 (m, k) 使得 v = m * 10^k 且 m ∈ [1, 10)。"""
    av = abs(v)
    k = math.floor(math.log10(av))
    m = av / (10 ** k)
    if m >= 10:
        m /= 10
        k += 1
    return m, k

def _sig_equal(a: float, b: float, sig_digits: int = 5) -> bool:
    """
    判断 a 与 b 的“前 sig_digits 位有效数字”是否一致。
    例如：123456000 与 1.23456 视为有效五位相同。
    """
    if a == 0.0 or b == 0.0:
        return False
    if math.copysign(1.0, a) != math.copysign(1.0, b):
        return False
    ma, _ = _mantissa_base10(a)
    mb, _ = _mantissa_base10(b)
    abs_tol = 0.5 * (10 ** (1 - sig_digits))
    return math.isclose(ma, mb, rel_tol=0.0, abs_tol=abs_tol)

def eval_model_result(
    success: bool,
    result: Any,
    ground_truth: Any,
    err_range: float = 1e-3,
    sig_digits: int = 5,
) -> Tuple[bool, bool]:
    """
    评测优化结果是否正确。

    ✅ 逻辑：
      1. 若 success=False，则返回 (False, False)
      2. 若 result 与 ground_truth 均包含“无解”标志 → 返回 (True, True)
      3. 否则：
         - 若两者数值近似 (math.isclose)，返回 (True, True)
         - 若单位不同但前 sig_digits 位一致，返回 (True, True)
         - 否则 (True, False)

    ✅ 参数：
      success: bool，表示模型是否正常运行
      result, ground_truth: 任意类型（list / int / float / str）
      err_range: float，math.isclose 的相对/绝对误差
      sig_digits: int，用于有效数字比较的位数

    ✅ 返回：
      (pass_flag, correct)
    """
    if not success:
        return False, False

    r_list = _to_list(result)
    g_list = _to_list(ground_truth)

    # 若双方均表明“无解”，则认为正确
    if any(_is_no_solution(r) for r in r_list) and any(_is_no_solution(g) for g in g_list):
        return True, True

    # 数值比较
    for r in r_list:
        rv = _to_float(r)
        if rv is None:
            continue
        for g in g_list:
            gv = _to_float(g)
            if gv is None:
                continue
            # 1) 小差异
            if math.isclose(rv, gv, rel_tol=err_range, abs_tol=err_range):
                return True, True
            # 2) 有效数字比较（单位差异）
            if _sig_equal(rv, gv, sig_digits=sig_digits):
                return True, True

    # 所有组合都未匹配
    return True, False

def get_entry(path, index):
    with open(path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    for i, entry in dataset.items():
        if i == str(index):
            return entry
    return None

#  -------------------async_extract_and_execute_python_code:判断答案ground_truth和result是否一致---------------------

# Extract and Execute:提取结果中所有的Python代码块，存入指定路径，并执行
async def async_extract_and_execute_python_code(
        text_content, entry, output_dir=None, attempt=0,
        timeout=DEFAULT_SUBPROCESS_TIMEOUT
):
    """
    从 text_content 中提取所有 ```python ... ``` 代码块，
    如果 output_dir 为 None，则使用临时文件执行；否则将代码块分别保存为
    output_dir/case_{entry['index']}.py 并执行。
    超过 timeout 秒会自动中断子进程并返回 False, "Timeout"。
    返回 (成功标志, 输出字符串 或 错误信息)。
    """
    # 提取所有 Python 代码块
    python_code_blocks = re.findall(r'```python\s*([\s\S]*?)```', text_content)

    if not python_code_blocks:
        print("未找到Python代码块。")
        return False, "No Python code blocks found"

    for code_block in python_code_blocks:
        code_block = code_block.strip()
        if not code_block:
            continue

        temp_file_path = None
        use_tempfile = (output_dir is None)

        try:
            if use_tempfile:
                # 如果没有指定 output_dir，则使用临时文件
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp_file:
                    tmp_file.write(code_block)
                    temp_file_path = tmp_file.name
            else:
                # 如果指定了 output_dir，则按照 case_{index}.py 保存在该目录
                os.makedirs(output_dir, exist_ok=True)
                filename = f"case_{entry['index']}_{attempt + 1}.py"
                temp_file_path = os.path.join(output_dir, filename)
                with open(temp_file_path, "w", encoding="utf-8") as f:
                    f.write(code_block)

            print(f"\n进入 await asyncio.create_subprocess_exec 执行：{temp_file_path}\n")
            proc = await asyncio.create_subprocess_exec(
                sys.executable, temp_file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                # 用 wait_for 给 communicate 加超时
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                # 超时，杀掉子进程并标记为失败
                proc.kill()
                await proc.wait()
                print(f"子进程超时（>{timeout}s），已被终止。")
                return False, f"Timeout after {timeout}s"

            if proc.returncode == 0:
                stdout_str = stdout.decode()
                # 从stdout_str中，使用extract_best_objective(stdout_str)提取唯一答案
                best_obj = extract_best_objective(stdout_str)
                return True, (best_obj if best_obj is not None else stdout_str)
            else:
                return False, stderr.decode()

        except Exception as e:
            return False, str(e)

        finally:
            # 只有在使用临时文件的情况下才删除
            if use_tempfile and temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    return False, "No valid code blocks executed"


if __name__ == '__main__':
    output = r'''
    E:\my_evns\py312_torch28\python.exe D:\\LLMProject\\NExTORAgent2025\\runs\\【20250605_163347】_【complexor】_【o4-mini-2025-04-16-high】\\case_6_1.py 
Restricted license - for non-production use only - expires 2026-11-23
Gurobi Optimizer version 12.0.2 build v12.0.2rc0 (win64 - Windows 11.0 (26100.2))

CPU model: Intel(R) Core(TM) Ultra 9 275HX, instruction set [SSE2|AVX|AVX2]
Thread count: 24 physical cores, 24 logical processors, using up to 24 threads

Optimize a model with 1 rows, 3 columns and 3 nonzeros
Model fingerprint: 0xb5b57700
Variable types: 0 continuous, 3 integer (3 binary)
Coefficient statistics:
  Matrix range     [1e+01, 3e+01]
  Objective range  [6e+01, 1e+02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [5e+01, 5e+01]
Found heuristic solution: objective 160.0000000
Presolve removed 1 rows and 3 columns
Presolve time: 0.00s
Presolve: All rows and columns removed

Explored 0 nodes (0 simplex iterations) in 0.00 seconds (0.00 work units)
Thread count was 1 (of 24 available processors)

Solution count 2: 220 160 

Optimal solution found (tolerance 1.00e-04)
Best objective 2.200000000000e+02, best bound 2.200000000000e+02, gap 0.0000%
Selected items:
  Item 1 (value=100, weight=20)
  Item 2 (value=120, weight=30)
Total value: 220.0
FinalAnswer=【220】

进程已结束，退出代码为 0
    '''
    output_F = r'''
    E:\my_evns\py312_torch28\python.exe D:\LLMProject\\NExTORAgent2025\\runs\【20250605_163347】_【complexor】_【o4-mini-2025-04-16-high】\case_7_2.py 
Restricted license - for non-production use only - expires 2026-11-23
Gurobi Optimizer version 12.0.2 build v12.0.2rc0 (win64 - Windows 11.0 (26100.2))

CPU model: Intel(R) Core(TM) Ultra 9 275HX, instruction set [SSE2|AVX|AVX2]
Thread count: 24 physical cores, 24 logical processors, using up to 24 threads

Optimize a model with 12 rows, 8 columns and 24 nonzeros
Model fingerprint: 0x013f1434
Variable types: 0 continuous, 8 integer (0 binary)
Coefficient statistics:
  Matrix range     [1e+00, 1e+00]
  Objective range  [1e+00, 4e+00]
  Bounds range     [0e+00, 0e+00]
  RHS range        [1e+01, 4e+01]
Presolve removed 9 rows and 8 columns
Presolve time: 0.00s

Explored 0 nodes (0 simplex iterations) in 0.00 seconds (0.00 work units)
Thread count was 1 (of 24 available processors)

Solution count 0

Model is infeasible
Best objective -, best bound -, gap -
No optimal solution found, cannot retrieve objective value.

进程已结束，退出代码为 0


    '''
    best_obj = extract_best_objective(output)
    print(best_obj)
    a,b = eval_model_result(
    True,
    best_obj,
    ["220",111],
    err_range = 1e-3,
    sig_digits= 5,
)
    print(a,b)
