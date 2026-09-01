import json
import sys
import asyncio
import tempfile
import os
from typing import Optional, Union, Any
import re
import math
from decimal import Decimal, InvalidOperation
from typing import List, Tuple

from openai import AsyncOpenAI

DEFAULT_SUBPROCESS_TIMEOUT = 200

_SOLVER_SUBPROCESS_ENV_KEYS = (
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROGRAMW6432",
    "HOMEDRIVE",
    "HOMEPATH",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "PYTHONPATH",
    "PYTHONUTF8",
    "PYTHONIOENCODING",
    "GRB_LICENSE_FILE",
    "GUROBI_HOME",
    "LANG",
    "LC_ALL",
    "TZ",
)


def build_solver_subprocess_env(source_env=None):
    """Return the minimal environment needed by local Python/Gurobi execution.

    LLM/API credentials, proxy variables, and cloud credentials are excluded by
    construction because only the explicit allowlist above is copied.
    """
    source = os.environ if source_env is None else source_env
    source_by_upper = {key.upper(): value for key, value in source.items()}
    child_env = {
        key: source_by_upper[key]
        for key in _SOLVER_SUBPROCESS_ENV_KEYS
        if key in source_by_upper
    }
    child_env.setdefault("PYTHONUTF8", "1")
    child_env.setdefault("PYTHONIOENCODING", "utf-8")
    return child_env

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
    r'\bFinalAnswer\b\s*=\s*【\s*([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)\s*】',
    re.IGNORECASE,
)
_RE_FINAL_MARKER = re.compile(r'\bFinalAnswer\b\s*=', re.IGNORECASE)
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

def extract_final_answer(output_text: str) -> List[str]:
    """
    严格提取唯一的标量 ``FinalAnswer=【...】``。

    无 marker、任一 marker 非法，或出现多个数值不同的 marker 时均拒绝；
    solver 日志中的 objective 不属于 benchmark answer，不在此函数中回退。
    """
    if not output_text:
        return ["None"]

    marker_count = len(list(_RE_FINAL_MARKER.finditer(output_text)))
    final_candidates = []
    for m in _RE_FINAL.finditer(output_text):
        v = _safe_float(m.group(1))
        try:
            decimal_value = Decimal(m.group(1))
        except InvalidOperation:
            decimal_value = None
        if v is not None and decimal_value is not None and decimal_value.is_finite():
            final_candidates.append((m.start(1), v, decimal_value))

    if marker_count == 0 or len(final_candidates) != marker_count:
        return ["None"]

    unique_values = {candidate[2].normalize() for candidate in final_candidates}
    if len(unique_values) != 1:
        return ["None"]

    final_candidates.sort(key=lambda candidate: candidate[0])
    return [f"{final_candidates[0][1]}"]


def extract_solver_objectives(output_text: str,
                              *,
                              include_best: bool = True,
                              rel_tol: float = 1e-9,
                              abs_tol: float = 1e-9) -> List[str]:
    """Extract solver-log objectives for diagnostics, never benchmark scoring."""
    if not output_text:
        return ["None"]

    candidates: List[Tuple[int, float]] = []
    for pattern in (_RE_OPTIMAL, _RE_BEST) if include_best else (_RE_OPTIMAL,):
        for match in pattern.finditer(output_text):
            value = _safe_float(match.group(1))
            if value is not None:
                candidates.append((match.start(1), value))

    candidates.sort(key=lambda kv: kv[0])
    values_in_order = [v for _, v in candidates]
    if not values_in_order:
        return ["None"]
    unique_vals = _dedup_with_tolerance(values_in_order, rel_tol=rel_tol, abs_tol=abs_tol)
    return [f"{v}" for v in unique_vals]


def extract_best_objective(output_text: str, **_unused) -> List[str]:
    """Backward-compatible name for the strict benchmark answer extractor."""
    return extract_final_answer(output_text)



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
    数量级必须一致；该检查只容忍相同数量级内的有效数字舍入误差。
    """
    if a == 0.0 or b == 0.0:
        return False
    if math.copysign(1.0, a) != math.copysign(1.0, b):
        return False
    ma, ka = _mantissa_base10(a)
    mb, kb = _mantissa_base10(b)
    if ka != kb:
        return False
    abs_tol = 0.5 * (10 ** (1 - sig_digits))
    return math.isclose(ma, mb, rel_tol=0.0, abs_tol=abs_tol)


def _reported_unit_tolerance(value: Any, floor: float) -> float:
    """Tolerance of one unit in the last ground-truth digit as reported."""
    try:
        decimal_value = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return floor
    if not decimal_value.is_finite():
        return floor
    exponent = decimal_value.as_tuple().exponent
    resolution = float(Decimal(1).scaleb(exponent)) if exponent < 0 else 1.0
    return max(floor, resolution * 1.000001)

def eval_model_result(
    success: bool,
    result: Any,
    ground_truth: Any,
    err_range: float = 1e-6,
    sig_digits: int = 5,
) -> Tuple[bool, bool]:
    """
    评测优化结果是否正确。

    ✅ 逻辑：
      1. 若 success=False，则返回 (False, False)
      2. 若 result 与 ground_truth 均包含“无解”标志 → 返回 (True, True)
      3. 否则：
         - 数值误差不超过 ground truth 已报告末位的一单位时，返回 (True, True)
         - 否则 (True, False)

    ✅ 参数：
      success: bool，表示模型是否正常运行
      result, ground_truth: 任意类型（list / int / float / str）
      err_range: float，缺少可用小数精度时的最小绝对容差
      sig_digits: 保留的兼容参数；新评分不再使用跨数量级有效数字回退

    ✅ 返回：
      (pass_flag, correct)
    """
    if not success:
        return False, False

    r_list = _to_list(result)
    g_list = _to_list(ground_truth)

    # Vector answers are positional. A scalar output must not receive credit by
    # matching just one element of a multi-value ground truth.
    if len(r_list) != len(g_list):
        return True, False

    for r, g in zip(r_list, g_list):
        r_missing = _is_no_solution(r)
        g_missing = _is_no_solution(g)
        if r_missing or g_missing:
            if r_missing and g_missing:
                continue
            return True, False

        rv = _to_float(r)
        gv = _to_float(g)
        if rv is None or gv is None:
            return True, False
        tolerance = _reported_unit_tolerance(g, floor=err_range)
        if math.isclose(rv, gv, rel_tol=0.0, abs_tol=tolerance):
            continue
        return True, False

    return True, True

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
                stderr=asyncio.subprocess.PIPE,
                env=build_solver_subprocess_env(),
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
                # 只接受显式且唯一的 FinalAnswer；solver objective 仅可单独诊断。
                best_obj = extract_final_answer(stdout_str)
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

