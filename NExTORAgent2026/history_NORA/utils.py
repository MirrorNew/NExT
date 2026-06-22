import re
import subprocess
import sys
import asyncio
import tempfile
import os
DEFAULT_SUBPROCESS_TIMEOUT = 200

def is_number_string(s):
    """
    Determine if a string is a numeric string, including:
    - Integers (e.g., "123", "-45")
    - Decimals (e.g., "3.14", "-0.001")
    - Scientific notation (e.g., "1.23e-4", "4.5E+6")

    Args:
        s (str): The string to be checked.

    Returns:
        bool: True if the string is a valid numeric string, otherwise False.
    """
    pattern = r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$"
    return re.fullmatch(pattern, s) is not None

def convert_to_number(s):
    """
    Convert a string to a number (integer or float).

    Args:
        s: The string to be converted.

    Returns:
        int or float: Returns int if the string represents an integer, float if it represents a decimal.
        Returns None if conversion fails.
    """
    try:
        # Try to convert to integer
        if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
            return int(s)
        # Try to convert to float
        num = float(s)
        return num
    except (ValueError, TypeError):
        return None

def extract_best_objective(output_text):
    """
    Extract Best objective or Optimal objective value from Gurobi output.
    
    Args:
        output_text: Gurobi output text
    
    Returns:
        float or None: Optimal solution value, returns None if not found
    """
    final_answer = []
    # print(f"FinalAnswer=【{the_question_answer}】")
    match_FinalAnswer = re.search(r'FinalAnswer=【([\d.eE+-]+)】', output_text)
    if match_FinalAnswer:
        the_question_answer = float(match_FinalAnswer.group(1))
        print(f"FinalAnswer=【{the_question_answer}】")
        final_answer.append(the_question_answer)

    # First check if model is infeasible
    if "Model is infeasible" in output_text:
        return None
    
    # Try to find Best objective
    match = re.search(r'Best objective\s+([\d.e+-]+)', output_text)
    if not match:
        # If not found, try to find Optimal objective
        match = re.search(r'Optimal objective\s+([\d.e+-]+)', output_text)

    if match:
        try:
            final_answer.append(float(match.group(1)))
            # return float(match.group(1))
            return final_answer
        except ValueError:
            return None

    if len(final_answer) > 0:
        return final_answer
    return None



def eval_model_result(success, result, ground_truth, err_range=0.001):

    # 在一定范围内，判断结果和真值是否相等
    def verify_result(calculated_value, true_value, epsilon=1e-3):
        """
        验证求解结果和真值是否相等

        参数:
            calculated_value: 计算得到的值
            true_value: 真实值
            epsilon: 对于接近值的相对误差容忍度(默认为0.001，即千分之一)

        返回:
            bool: 如果满足条件返回True，否则返回False
        """
        try:
            # 处理除数为0的情况
            if true_value == 0:
                return calculated_value - 0 < epsilon

            ratio = abs(calculated_value / true_value)

            # 情况1: 两个数值差距很大(相差十倍以外)
            if ratio > 10 or ratio < 0.1:
                def get_significant_digits(x):
                    # 将数值转换为字符串，去除负号
                    s = str(abs(x))

                    # 移除小数点
                    s = s.replace('.', '')

                    # 去除前导零，保留有效数字
                    significant_digits = s.lstrip('0')

                    # 如果所有数字都是0（如0.000），返回'0'
                    if not significant_digits:
                        return '0'

                    return significant_digits

                # 获取两个数的有效数字
                sig_calc = get_significant_digits(calculated_value)
                sig_true = get_significant_digits(true_value)

                # 取前五位有效数字进行比较
                min_len = min(5, len(sig_calc), len(sig_true))
                return sig_calc[:min_len] == sig_true[:min_len]

            # 情况2: 两个数值接近(相差十倍以内)
            else:
                relative_error = abs((calculated_value - true_value) / true_value)
                return relative_error <= epsilon

        except Exception as e:
            print(f"验证过程中发生错误: {e}")
            return False

    pass_flag = False
    correct_flag = False
    if success:
        pass_flag = True
        result = result if isinstance(result,list) else [result]
        ground_truth = ground_truth if isinstance(ground_truth,list) else [ground_truth]
        for i in range(len(result)):
            # 只要有一个对了，就认为对了，打破循环
            if correct_flag: break
            for j in range(len(ground_truth)):
                if is_number_string(str(result[i])) and ground_truth[j] is not None:
                    result_num = convert_to_number(str(result[i]))
                    ground_truth_num = convert_to_number(str(ground_truth[j]))
                    correct_flag = verify_result(result_num, ground_truth_num, epsilon=1e-3)
                    # 只要有一个对了，就认为对了，打破循环
                    if correct_flag: break
                elif result[i] is None or result[i] == 'None' : # no available solution
                    if ground_truth[j] is None or ground_truth[j] == 'None':
                        correct_flag = True
    return pass_flag, correct_flag



# Extract and Execute:提取结果中所有的Python代码块，存入指定路径，并执行
async def async_extract_and_execute_python_code(
        text_content, entry, output_dir=None, attempt=1,
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


# 测试示例
# if __name__ == "__main__":
#     # 测试情况1: 数值差距很大(单位不同)
#     print(verify_result(1234567, 1234500))  # True (有效数字前五位相同)
#     print(verify_result(0.00123456, 0.00123400))  # True (有效数字前五位相同)
#     print(verify_result(0.00012345, 0.00012340))  # True (有效数字前五位相同)
#     print(verify_result(0.00012345, 0.00012355))  # False (第五位有效数字不同)
#     print(verify_result(0.023, 0.0234))  # True (有效数字"23"相同)
#     print(verify_result(0.023, 0.024))  # False (有效数字第一位不同)
#
#     # 测试情况2: 数值接近
#     print(verify_result(100.1, 100.0))  # True (误差0.1%)
#     print(verify_result(0.01001, 0.01))  # True (误差0.1%)
#     print(verify_result(0.01002, 0.01))  # False (误差0.2% > 0.1%)
#     print(verify_result(9984.22715, 9984.13))  # True
#
#     # 测试边界情况
#     print(verify_result(4.201780957396e-07, 0.0))  # True
#     print(verify_result(0.000, 0.000))  # True
#     a = [
#       23.0,
#       4.201780957396e-07
#     ]
#
#     b = [
#       34.0,
#       0.0
#     ]
#
#     pass_flag, correct_flag = eval_model_result(True, a, b, err_range=0.001)
#
#     print(correct_flag)
# if __name__ == '__main__':
#     solve = True
#     value = None
#     ground_truth = None
#     eval_model_result(solve, value, ground_truth, err_range=0.1)



if __name__ == '__main__':
    output = '''
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
    output_F = '''
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
    best_obj = extract_best_objective(output_F)
    print(best_obj is None)