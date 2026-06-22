import asyncio
import json


from RebuildNORA_utils import get_entry, async_extract_and_execute_python_code
from agents.base_agents import BaseAgent
from test_for_rebuild_constr.temp import code_25


class RepairAgentCode(BaseAgent):
    def __init__(self, client, model_name="o3-mini", temperature=0.2):
        super().__init__(client, model_name, temperature)
        self.system_msg = (
            "You are an expert in Gurobi code debugging."
            "Diagnose problems and identify areas for improvement based on user inquiries, mathematical models, and error messages during code execution."
            "No need to provide complete code."
        )
        self.messages.append({"role": "system", "content": self.system_msg})

    async def generate(self, entry: dict, math_model: str, error_msg: str, gurobi_code: str) -> str:
        context = json.dumps({
            'variables': entry.get('Variables_List', []),
            'constraints': entry.get('Constraint_Table', []),
            'objective': entry.get('Objective', {}),
            'problem_type': entry.get('Problem_Type', ''),
        }, ensure_ascii=False)

        msgs = [
            {"role": "user", "content": f"Problem: {entry['question']}\n"
                                        f"Context: {context}"},
            {"role": "assistant", "content": f"Math model and coding advice:\n{math_model}\n"
                                             f"Current code:\n{gurobi_code}"},
            {"role": "user", "content": (
                f"Error encountered:\n{error_msg}\n"
                "Please suggest fixes, no need for full code."
            )}
        ]
        self.messages.extend(msgs)
        return await self._query()



class RepairAgent(BaseAgent):
    """
    RepairAgent implements a two-step, code-first repair workflow:

    Step 1 (analysis):
        - Given the original natural-language problem description (entry["question"]),
          the A2 extraction results (Variables_List / Parameters_List / Constraint_Table / Objective),
          and the current (possibly wrong) Gurobi code, split the code into logical blocks:
              * Variables_List   : all decision variable declarations
              * Parameters_List  : parameter / data definitions (if they appear in code)
              * Constraint_List  : each constraint (or group of constraints) as one block
              * Objective        : the objective definition block

        - For each code block, perform a code-first reverse translation:
              * infer a canonical math expression (math)
              * give a one-sentence natural language description (one_sentence_nl)
              * match it to 0..k original sentences (by sid)
              * decide: 'consistent' / 'need_fix' / 'spurious'
              * if 'need_fix', propose a corrected math expression (suggested_fix_math)

        - Output a JSON object that includes:
              * Original_Sentences : list of {sid, text}
              * code_info_list     : grouped code blocks
              * Model_Items        : analysis per block
              * Reconstructed_Paragraph : stitched description of the model implied by the code
              * Global_Comments    : short summary of main discrepancies

    Step 2 (code update):
        - Based on the structured JSON analysis from Step 1 and the original code (plus error_msg),
          generate an UPDATED Gurobi code snippet that:
              * applies all necessary fixes for items marked as 'need_fix' or 'spurious'
              * keeps unrelated code as close as possible to the original
              * is syntactically correct and logically consistent with the original problem text.
    """

    def __init__(self, client, model_name: str = "o3-mini", temperature: float = 0.2):
        super().__init__(client, model_name, temperature)
        # We always reset self.messages in each top-level call, so here we just store the system message.
        self.system_msg = (
            "You are an expert in optimization modeling and Gurobi code repair.\n"
            "You follow a code-first workflow:\n"
            "  (1) Split the Gurobi code into logical blocks (variables, parameters, constraints, objective),\n"
            "      reverse-translate each block into math + one-sentence NL, and compare it against the\n"
            "      original optimization problem description sentence by sentence, labeling each block as\n"
            "      'consistent', 'need_fix', or 'spurious'.\n"
            "  (2) Then, based on this structured analysis, you produce an updated Gurobi code that applies\n"
            "      all necessary fixes while keeping unrelated parts as unchanged as possible.\n"
            "The original natural-language problem text is the final source of truth; A2 extraction results\n"
            "are helpful hints (aliases, naming), but NOT the judge of correctness."
        )
        # ensure messages list exists
        self.messages = []

    # ===================== public main entry =====================

    async def forward(
        self,
        entry: dict,
        error_msg: str,
        gurobi_code: str,
        math_model: str = ""
    ):
        """
        Full repair pipeline.

        Args:
            entry: dict containing the problem description and A2 outputs, e.g.:
                   - entry["question"]: original natural-language description
                   - entry["Variables_List"], entry["Parameters_List"],
                     entry["Constraint_Table"], entry["Objective"], entry["Problem_Type"]
            error_msg: solver/runtime error message string（可能为空）
            gurobi_code: current (possibly wrong) Gurobi code as a string
            math_model: optional textual math-model draft (only used as hint)

        Returns:
            analysis_json_str: str, JSON string returned by step 1
            updated_code_str:  str, updated Gurobi code returned by step 2
        """
        analysis_json_str = await self._step1_analyze(entry, gurobi_code, math_model)
        updated_code_str = await self._step2_update_code(entry, error_msg, gurobi_code, analysis_json_str)
        return analysis_json_str, updated_code_str

    # ===================== step 1: code-first reverse & compare =====================

    async def _step1_analyze(self, entry: dict, gurobi_code: str, math_model: str = "") -> str:
        """
        Step 1:
          - Split code into code_info_list blocks.
          - Reverse-translate each block into (kind, name, math, one_sentence_nl, ...).
          - Match each block to 0..k original sentences and decide:
              'consistent' / 'need_fix' / 'spurious'.
          - Produce a JSON object with:
              {
                "Original_Sentences": [...],
                "code_info_list": {
                    "Variables_List": [...],
                    "Parameters_List": [...],
                    "Constraint_List": [...],
                    "Objective": "<code>"
                },
                "Model_Items": [...],
                "Reconstructed_Paragraph": "...",
                "Global_Comments": "..."
              }
        """
        original_text = entry.get("question", "")

        # A2 extraction is only used as hints (names, indices, related sentence ids).
        a2_hints = json.dumps(
            {
                "Variables_List": entry.get("Variables_List", []),
                "Parameters_List": entry.get("Parameters_List", []),
                "Constraint_Table": entry.get("Constraint_Table", []),
                "Objective": entry.get("Objective", {}),
                "Problem_Type": entry.get("Problem_Type", ""),
            },
            ensure_ascii=False,
        )

        # reset conversation for step 1
        self.messages = [{"role": "system", "content": self.system_msg}]

        user_prompt = f"""
我们正在调试一个运筹优化建模的 Gurobi 代码，请你按照“代码优先（code-first）”的思路，分两大步完成 **分析阶段**：

【已知信息】：
1. 原始题目描述（英文或中文）：
<<<
{original_text}
>>>

2. A2 阶段的提取结果（仅作为 alias / 命名线索，不作为裁判）：
<<<
{a2_hints}
>>>

3. 当前（可能有建模错误的）Gurobi 代码：
<<<
{gurobi_code}
>>>

4. 可选的数学模型草案（仅供参考，可为空）：
<<<
{math_model}
>>>

现在只做“分析”，暂时不要给出新的完整代码。请严格按如下步骤输出一个 JSON 对象：

Step 1：代码分块（code_info_list）
--------------------------------
请你将 Gurobi 代码按照信息来源分块，得到一个 code_info_list，对应四类：

- Variables_List   : 决策变量相关的代码块，每个元素是一段代码字符串，可以多行；
- Parameters_List  : 参数 / 数据的定义、赋值或读取片段（如果出现在该文件中）；
- Constraint_List  : 每一条（或一组逻辑相关的）约束作为一个代码块；
- Objective        : 目标函数设置的代码块（通常是 setObjective / setObjectiveN 一段）。

示例结构：
{{
  "Variables_List": [
    "<var_code_block_1>",
    "<var_code_block_2>"
  ],
  "Parameters_List": [
    "<param_code_block_1>"
  ],
  "Constraint_List": [
    "<constr_code_block_1>",
    "<constr_code_block_2>"
  ],
  "Objective": "<objective_code_block>"
}}

Step 2：逐块 Code-first 反译 + 与原题逐句对比
---------------------------------------------
1）先对原题进行分句，得到 Original_Sentences 列表，每一个元素形如：
    {{
      "sid": 1,
      "text": "<原题的第 1 句，可以仅写开头前几个单词，不需要写完整>"
    }}

2）针对 code_info_list 中的每一个代码块，执行 **Code-first 反译**，将其变成一个 Model_Items 元素，结构如下：
    {{
      "kind": "variable" | "parameter" | "constraint" | "objective",
      "name": "<代码中的核心名称，例如 x, y_ij, DemandCap_t>",
      "math": "<规范数学形式（带量词和索引，例如：sum_i x[i,t] <= 1.25 * base[i,t], ∀t∈T)>",
      "one_sentence_nl": "<一句话自然语言解释，贴近期末竞赛/教材风格>",
      "matched_sids": [<与该信息最相关的原题句子 sid 列表，可以为空>],
      "status": "consistent" | "need_fix" | "spurious",
      "issues": [
          "Quantifier_Mismatch" | "Bound_Direction_Mismatch" |
          "Indexing_Mismatch" | "Coefficient_Mismatch" |
          "Unit_Dimension_Mismatch" | "Soft_vs_Hard_Misuse" |
          "Missing_in_Text" | "Other"
      ],
      "suggested_fix_math": "<若 status='need_fix'，请给出建议的规范数学表达；否则写空字符串>"
    }}

说明：
- 你可以利用 A2 里的名称、句子编号等信息作为“提示”，帮助你更好匹配原题中的句子，但最终裁判标准必须是原题文字本身。
- 假设题目本身没有漏掉关键约束，因此如果代码中的某个块在原题中找不到语义支撑，请标记为 status = "spurious"，issues 中包含 "Missing_in_Text"。
- 如果某个块的数学含义与原题表达不一致（例如整体约束被误写成逐类约束），请标记为 "need_fix"，并在 suggested_fix_math 中写出正确的数学表达。

Step 3：整体重述
----------------
- 将所有 Model_Items 的 one_sentence_nl 按合理顺序串成一段话，写入 Reconstructed_Paragraph；
- 给出 Global_Comments，对照原题，总结主要差异：哪些信息缺失、哪些信息多余或口径错误。

【必须输出的 JSON 结构】（示例）：
{{
  "Original_Sentences": [
    {{"sid": 1, "text": "..."}},
    {{"sid": 2, "text": "..."}}
  ],
  "code_info_list": {{
    "Variables_List": ["<code1>", "<code2>",...],
    "Parameters_List": ["<code1>","<code2>",...],
    "Constraint_List": ["<code1>", "<code2>",...],
    "Objective": "<code>"
  }},
  "Model_Items": [
    {{
      "kind": "constraint",
      "name": "GrowthCap",
      "math": "sum_i x[i,t] <= 1.25 * base[i,t], ∀t∈T",
      "one_sentence_nl": "对每个时期 t，各产品总产量不超过该期基线销量的 1.25 倍。",
      "matched_sids": [7, 9],
      "status": "need_fix",
      "issues": ["Quantifier_Mismatch"],
      "suggested_fix_math": "sum_{{i,t}} x[i,t] <= 1.25 * sum_{{i,t}} base[i,t]"
    }}
  ],
  "Reconstructed_Paragraph": "<由 one_sentence_nl 串接而成的一段话>",
  "Global_Comments": "<对整体差异的简要说明>"
}}

请只输出一个合法的 JSON 对象，不要使用 Markdown 代码块，也不要添加任何额外说明文字。
"""
        self.messages.append({"role": "user", "content": user_prompt})
        result = await self._query()
        return result

    # ===================== step 2: apply fixes and generate updated code =====================

    async def _step2_update_code(
        self,
        entry: dict,
        error_msg: str,
        original_code: str,
        analysis_json_str: str,
    ) -> str:
        """
        Step 2:
          - Based on the analysis JSON from step 1 (which includes code_info_list and Model_Items)
            and the original Gurobi code (plus error_msg), generate an UPDATED code snippet.
          - The updated code should:
              * fix all items whose status == 'need_fix' (according to suggested_fix_math),
              * remove or weaken items marked as 'spurious' when they truly have no textual support,
              * keep unrelated parts of the code as unchanged as possible,
              * be a complete, syntactically correct Python + Gurobi model.

        Returns:
            updated_code: str, the new code (no extra commentary).
        """
        original_text = entry.get("question", "")

        # For step 2 we start a fresh conversation but include the step1 JSON as context in the user message.
        self.messages = [{"role": "system", "content": self.system_msg}]

        user_prompt = f"""
下面是你在 Step 1 中输出的分析结果 JSON（包含 Original_Sentences、code_info_list 和 Model_Items）：
<<<
{analysis_json_str}
>>>

[原始题目描述]（用于确认语义）：
<<<
{original_text}
>>>

[原始 Gurobi 代码]（需要在此基础上进行修改）：
<<<
{original_code}
>>>

[求解或运行时的错误信息]（可为空，仅作辅助参考）：
<<<
{error_msg}
>>>

请你根据上述信息，生成一份 **更新后的 Gurobi 建模代码**，要求：

1. 对于 Model_Items 中 status = "need_fix" 的条目：
   - 根据 suggested_fix_math 调整对应的 Gurobi 代码（例如修改 addConstr / setObjective / 变量定义等），
     使其数学含义与原题描述保持一致。
2. 对于 status = "spurious" 的条目：
   - 若该信息确实在原题找不到语义支撑，应删除对应的代码块，或酌情弱化（例如从硬约束改为注释或软约束），
     具体依据你在分析 JSON 中的判断。
3. 其他 status = "consistent" 的部分尽量保持不变。
4. 输出的代码应为一段 **完整的、可运行的 Python + Gurobi 代码**：
   - 包含模型定义、变量、约束、目标函数等必要部分；
   - 不需要包含 if __name__ == "__main__": 之类的主程序包装；
   - 语法必须正确、缩进合理。

请只输出新的 Python 代码正文，不要任何解释性文字，不要包裹在 Markdown 代码块中。
"""
        self.messages.append({"role": "user", "content": user_prompt})
        updated_code = await self._query()
        return updated_code

    # ===================== optional: keep old generate for compatibility =====================

    async def generate(
        self,
        entry: dict,
        math_model: str,
        error_msg: str,
        gurobi_code: str,
    ) -> str:
        """
        Backward-compatible simple interface:
          - If you still call `generate` elsewhere, this will run ONLY step 1 and
            return the analysis JSON string (without applying fixes).
          - For the full repair pipeline (analysis + updated code), please call `forward`.
        """
        analysis_json_str = await self._step1_analyze(entry, gurobi_code, math_model)
        return analysis_json_str


import os
import openai
from dotenv import load_dotenv
# ---------------- 1.1 环境初始化 -------------------------------------------------
load_dotenv()
# ---------------- 1.2 API 客户端初始化 -------------------------------------------
openai_api_data = dict(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)
if __name__ == '__main__':
    async_openai = openai.AsyncOpenAI(
        api_key=openai_api_data['api_key'],
        base_url=openai_api_data['base_url'] or None
    )

    repair_agent = RepairAgent(async_openai, model_name='o4-mini')
    cu = '''\'\'\'python import gurobipy as gp

# 1. Parameters (from the provided list)
octane_naphtha = {'light': 90, 'medium': 80, 'heavy': 70}
octane_reformed_gasoline = 115
yield_reform_gas_light_naphtha = 0.6
yield_reform_gas_medium_naphtha = 0.52
yield_reform_gas_heavy_naphtha = 0.45
octane_cracking_gasoline = 105
yield_pyrolysis_oil_light = 0.68
yield_pyrolysis_gasoline_light = 0.28
yield_pyrolysis_oil_heavy = 0.75
yield_pyrolysis_gasoline_heavy = 0.2
yield_lubricating_from_residual = 0.5
min_octane_premium_engine_oil = 94
min_octane_ordinary_engine_oil = 84
max_pressure_kerosene = 1.0
pressure_light_oil = 1.0
pressure_heavy_oil = 0.6
pressure_pyrolysis_oil = 1.5
pressure_residual_oil = 0.05
ratio_fuel_oil = {'light_oil': 10, 'heavy_oil': 3, 'cracking_oil': 4, 'residual_oil': 1}
avail_crude_oil_1 = 20000
avail_crude_oil_2 = 30000
cap_distillation = 45000
cap_reforming = 10000
cap_cracking = 8000
lubricating_oil_min = 500
lubricating_oil_max = 1000
min_ratio_premium_to_ordinary = 0.4
profit_premium_engine_oil = 700
profit_ordinary_engine_oil = 600
profit_kerosene = 400
profit_fuel_oil = 350
profit_lubricating_oil = 150
Table_1_C_1 = {
    'Crude oil 1': {'Light naphtha': 0.1, 'Medium naphtha': 0.2, 'Heavy naphtha': 0.2,
                    'Light oil': 0.12, 'Heavy oil': 0.2, 'Residue': 0.13},
    'Crude oil 2': {'Light naphtha': 0.15, 'Medium naphtha': 0.25, 'Heavy naphtha': 0.18,
                    'Light oil': 0.08, 'Heavy oil': 0.19, 'Residue': 0.12}
}

# 2. Create model
model = gp.Model("XXXXXXXX")

# 3. Decision variables
D1 = model.addVar(lb=0, ub=avail_crude_oil_1, name="D1")
D2 = model.addVar(lb=0, ub=avail_crude_oil_2, name="D2")

N_L = model.addVar(lb=0, name="N_L")
N_M = model.addVar(lb=0, name="N_M")
N_H = model.addVar(lb=0, name="N_H")
LO  = model.addVar(lb=0, name="LO")
HO  = model.addVar(lb=0, name="HO")
R   = model.addVar(lb=0, name="R")

N_L_to_mix    = model.addVar(lb=0, name="N_L_to_mix")
N_L_to_reform = model.addVar(lb=0, name="N_L_to_reform")
N_M_to_mix    = model.addVar(lb=0, name="N_M_to_mix")
N_M_to_reform = model.addVar(lb=0, name="N_M_to_reform")
N_H_to_mix    = model.addVar(lb=0, name="N_H_to_mix")
N_H_to_reform = model.addVar(lb=0, name="N_H_to_reform")

RG_L = model.addVar(lb=0, name="RG_L")
RG_M = model.addVar(lb=0, name="RG_M")
RG_H = model.addVar(lb=0, name="RG_H")

LO_to_mix   = model.addVar(lb=0, name="LO_to_mix")
LO_to_crack = model.addVar(lb=0, name="LO_to_crack")
HO_to_mix   = model.addVar(lb=0, name="HO_to_mix")
HO_to_crack = model.addVar(lb=0, name="HO_to_crack")

CO_LO = model.addVar(lb=0, name="CO_LO")
CG_LO = model.addVar(lb=0, name="CG_LO")
CO_HO = model.addVar(lb=0, name="CO_HO")
CG_HO = model.addVar(lb=0, name="CG_HO")
CO    = model.addVar(lb=0, name="CO")
CG    = model.addVar(lb=0, name="CG")

R_to_mix = model.addVar(lb=0, name="R_to_mix")
R_to_lub = model.addVar(lb=0, name="R_to_lub")
Lub      = model.addVar(lb=0, name="Lub")

N_L_H  = model.addVar(lb=0, name="N_L_H")
N_M_H  = model.addVar(lb=0, name="N_M_H")
N_H_H  = model.addVar(lb=0, name="N_H_H")
N_L_O  = model.addVar(lb=0, name="N_L_O")
N_M_O  = model.addVar(lb=0, name="N_M_O")
N_H_O  = model.addVar(lb=0, name="N_H_O")
RG_L_H = model.addVar(lb=0, name="RG_L_H")
RG_M_H = model.addVar(lb=0, name="RG_M_H")
RG_H_H = model.addVar(lb=0, name="RG_H_H")
RG_L_O = model.addVar(lb=0, name="RG_L_O")
RG_M_O = model.addVar(lb=0, name="RG_M_O")
RG_H_O = model.addVar(lb=0, name="RG_H_O")
CG_H   = model.addVar(lb=0, name="CG_H")
CG_O   = model.addVar(lb=0, name="CG_O")

HE = model.addVar(lb=0, name="HE")
OE = model.addVar(lb=0, name="OE")
K  = model.addVar(lb=0, name="K")
FO = model.addVar(lb=0, name="FO")

# 5. Objective
model.setObjective(
      profit_premium_engine_oil * HE
    + profit_ordinary_engine_oil * OE
    + profit_kerosene * K
    + profit_fuel_oil * FO
    + profit_lubricating_oil * Lub,
    gp.GRB.MAXIMIZE
)

# 6. Constraints
model.addConstr(N_L == Table_1_C_1['Crude oil 1']['Light naphtha']  * D1
                   + Table_1_C_1['Crude oil 2']['Light naphtha']  * D2, name="C1")
model.addConstr(N_M == Table_1_C_1['Crude oil 1']['Medium naphtha']* D1
                   + Table_1_C_1['Crude oil 2']['Medium naphtha']* D2, name="C2")
model.addConstr(N_H == Table_1_C_1['Crude oil 1']['Heavy naphtha'] * D1
                   + Table_1_C_1['Crude oil 2']['Heavy naphtha'] * D2, name="C3")
model.addConstr(LO  == Table_1_C_1['Crude oil 1']['Light oil']     * D1
                   + Table_1_C_1['Crude oil 2']['Light oil']     * D2, name="C4")
model.addConstr(HO  == Table_1_C_1['Crude oil 1']['Heavy oil']     * D1
                   + Table_1_C_1['Crude oil 2']['Heavy oil']     * D2, name="C5")
model.addConstr(R   == Table_1_C_1['Crude oil 1']['Residue']       * D1
                   + Table_1_C_1['Crude oil 2']['Residue']       * D2, name="C6")

model.addConstr(N_L_to_mix    + N_L_to_reform    == N_L, name="C7")
model.addConstr(N_M_to_mix    + N_M_to_reform    == N_M, name="C8")
model.addConstr(N_H_to_mix    + N_H_to_reform    == N_H, name="C9")

model.addConstr(RG_L == yield_reform_gas_light_naphtha  * N_L_to_reform, name="C10")
model.addConstr(RG_M == yield_reform_gas_medium_naphtha * N_M_to_reform, name="C11")
model.addConstr(RG_H == yield_reform_gas_heavy_naphtha  * N_H_to_reform, name="C12")

model.addConstr(LO_to_mix   + LO_to_crack   == LO, name="C13")
model.addConstr(HO_to_mix   + HO_to_crack   == HO, name="C14")

model.addConstr(CO_LO == yield_pyrolysis_oil_light      * LO_to_crack, name="C15")
model.addConstr(CG_LO == yield_pyrolysis_gasoline_light * LO_to_crack, name="C16")
model.addConstr(CO_HO == yield_pyrolysis_oil_heavy      * HO_to_crack, name="C17")
model.addConstr(CG_HO == yield_pyrolysis_gasoline_heavy * HO_to_crack, name="C18")
model.addConstr(CO    == CO_LO + CO_HO, name="C19")
model.addConstr(CG    == CG_LO + CG_HO, name="C20")

model.addConstr(R_to_mix + R_to_lub == R, name="C21")
model.addConstr(Lub == yield_lubricating_from_residual * R_to_lub, name="C22")

model.addConstr(N_L_H  + N_L_O  == N_L_to_mix, name="C23")
model.addConstr(N_M_H  + N_M_O  == N_M_to_mix, name="C24")
model.addConstr(N_H_H  + N_H_O  == N_H_to_mix, name="C25")
model.addConstr(RG_L_H + RG_L_O == RG_L,      name="C26")
model.addConstr(RG_M_H + RG_M_O == RG_M,      name="C27")
model.addConstr(RG_H_H + RG_H_O == RG_H,      name="C28")
model.addConstr(CG_H   + CG_O   == CG,        name="C29")

model.addConstr(HE == N_L_H + N_M_H + N_H_H + RG_L_H + RG_M_H + RG_H_H + CG_H, name="C30")
model.addConstr(OE == N_L_O + N_M_O + N_H_O + RG_L_O + RG_M_O + RG_H_O + CG_O, name="C31")

model.addConstr(
    octane_naphtha['light']  * N_L_H
  + octane_naphtha['medium'] * N_M_H
  + octane_naphtha['heavy']  * N_H_H
  + octane_reformed_gasoline * (RG_L_H + RG_M_H + RG_H_H)
  + octane_cracking_gasoline * CG_H
  >= min_octane_premium_engine_oil * HE,
  name="C32"
)
model.addConstr(
    octane_naphtha['light']  * N_L_O
  + octane_naphtha['medium'] * N_M_O
  + octane_naphtha['heavy']  * N_H_O
  + octane_reformed_gasoline * (RG_L_O + RG_M_O + RG_H_O)
  + octane_cracking_gasoline * CG_O
  >= min_octane_ordinary_engine_oil * OE,
  name="C33"
)

model.addConstr(
    pressure_light_oil    * LO_to_mix
  + pressure_heavy_oil    * HO_to_mix
  + pressure_pyrolysis_oil * CO
  + pressure_residual_oil  * R_to_mix
  <= LO_to_mix,
  name="C34"
)

model.addConstr(
    ratio_fuel_oil['heavy_oil']   * LO_to_mix
  == ratio_fuel_oil['light_oil']   * HO_to_mix,
  name="C35"
)
model.addConstr(
    ratio_fuel_oil['cracking_oil'] * HO_to_mix
  == ratio_fuel_oil['heavy_oil']   * CO,
  name="C36"
)
model.addConstr(
    LO_to_mix
  == ratio_fuel_oil['light_oil']  * R_to_mix,
  name="C37"
)

model.addConstr(D1 + D2 <= cap_distillation, name="C40")
model.addConstr(N_L_to_reform + N_M_to_reform + N_H_to_reform <= cap_reforming, name="C41")
model.addConstr(LO_to_crack + HO_to_crack <= cap_cracking, name="C42")

model.addConstr(Lub >= lubricating_oil_min, name="C43_min")
model.addConstr(Lub <= lubricating_oil_max, name="C43_max")

model.addConstr(HE >= min_ratio_premium_to_ordinary * OE, name="C44")

model.addConstr(K + FO == LO_to_mix + HO_to_mix + CO + R_to_mix, name="C45")

# 7. Solve and output
model.optimize()
print(f"FinalAnswer=【{model.objVal}】")
    \'\'\''''
    entry = get_entry('NORA_process_data/NExTLP_SbS_o4mini.json',25)

    analysis_json_str, repaired_code = asyncio.run( repair_agent.forward(
        entry=entry,
        error_msg="答案错误,现有代码计算出来为：Infeasible or unbounded model，实际有解，为40.25",
        gurobi_code=code_25,
        math_model="",  # 可选，不需要就传 ""
    ))
    print(f"analysis_json_str={analysis_json_str}")
    print(f"repaired_code={repaired_code}")

    repaired_code = '```python '+repaired_code+'```'
    ok,result = asyncio.run(async_extract_and_execute_python_code(text_content=repaired_code,
                                          entry=entry,
                                          output_dir="runs_test_for_repair_agents"
                                          ))
    print(f"ok={ok}, result={result}")