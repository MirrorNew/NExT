# -*- coding: utf-8 -*-
"""
Async version of or_llm_eval_multi_agents.py with agent classes for modeling, coding, and repair using openai.AsyncOpenAI.
"""
import datetime
import os
import re
import sys
import json
import argparse
import tempfile
import asyncio
from tqdm import tqdm
import re

from agents.ExtractMultiParameterFromLongTextAgent import ExtractMultiParameterFromLongTextAgent


def quote_value_section(s: str) -> str:
    # 先匹配 "Value": { ... } 整块
    pattern = re.compile(r'("Value"\s*:\s*\{)([^}]*)(\})')
    def _repl(m):
        head, body, tail = m.group(1), m.group(2), m.group(3)
        # 再在 body 里把所有脱引号数字 key 加上双引号
        body_fixed = re.sub(r'(\b\d+\b)\s*:', r'"\1":', body)
        return head + body_fixed + tail

    return pattern.sub(_repl, s)

def fix_value_section(s: str) -> str:
    # 1) 可选：去掉 markdown 代码块标记
    s = re.sub(r'```(?:json)?\s*|\s*```', '', s)

    # 2) 先把整个 "Value": { … } 区域重抽出来、逐块处理
    def repl(m):
        head, body, tail = m.group(1), m.group(2), m.group(3)
        # 2.1) 给所有脱引号的 key 自动补上双引号（如果还有下层数字 key）
        body = re.sub(r'([{,]\s*)(\w+)\s*:', r'\1"\2":', body)
        # 2.2) 把所有 '(x, y)' → '[x, y]'
        body = re.sub(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', r'[\1,\2]', body)
        return head + body + tail

    pattern = re.compile(r'("Value"\s*:\s*\{)(.*?)(\})', re.S)
    return pattern.sub(repl, s)


# 用法示例
# raw = '"Value": {0: 163, 1: 180, 2: 208}, "Foo": {"bar":1}'
# fixed = quote_value_section(raw)
# print(fixed)
# # -> '"Value": {"0": 163, "1": 180, "2": 208}, "Foo": {"bar":1}'


import copy
DEFAULT_SUBPROCESS_TIMEOUT = 200
import openai
from dotenv import load_dotenv

from utils import (
    is_number_string,
    convert_to_number,
    extract_best_objective,
    eval_model_result, eval_model_result_origin
)

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



# ---------------- 2. Agent Classes START --------------------------------------------
class BaseAgent:
    def __init__(self, client, model_name="o3-mini", temperature=0.2):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.messages = []

    async def _query(self):
        resp = await self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            temperature=self.temperature
        )
        content = resp.choices[0].message.content
        self.messages.append({"role": "assistant", "content": content})
        return content

class Simple_agent(BaseAgent):
    def __init__(self, client, model_name="o3-mini", temperature=0.2):
        super().__init__(client, model_name, temperature)
        self.system_msg = (
            '''
           You are an operations optimization expert. 
           Please construct a mathematical model based on the operational optimization problem provided by the user, and write complete and reliable Python code to solve the operational optimization problem using Gurobi. 
           Please include necessary model construction, variable definition, constraint addition, objective function setting, solution, and result output in the code. 
           Output in the form of ```python\n{code}\n```, without the need for code description.
            '''
        )
        self.messages.append({"role": "system", "content": self.system_msg})

    async def generate(self, entry: dict) -> str:
        self.messages.extend([{"role": "user", "content": f"Problem: {entry['question']}\n"}])
        return await self._query()


# ---------------- 2.1 FactorsFromLongTextAgent  --------------------------------------------
class ExtractFactorsFromLongTextAgent(BaseAgent):
    def __init__(self, client, model_name="o3-mini", temperature=0.2):
        super().__init__(client, model_name, temperature)
        self.system_msg = (
            '''
            You are an expert mathematical modeler and an optimization professor at a top university. We will give you a description of an optimization problem.
            Regarding this combinatorial optimization problem, please complete the following tasks:
            Extract all decision variables and constraints from the following paragraph, ensuring that no element from the sentences is overlooked.
            
            1. **Sentence Scanning**: Start by providing the original sentence number and content, and then scan sentence by sentence. EITHER extract it into one or more constraints information OR mark it as "No constraints".
            2. **Variable List**: Give Variables from constraints sentence, and point: Name(symbol) / Meaning / type:<integer type OR continuous type>" / Range of Values. 
            3. **Mapping Table**: In a Markdown table, precisely correspond the "Constraint Name ↔ Mathematical Expression ↔ Sentence Number."
            4. **Optimization Goal**: Provide the optimization objective (target or performance metric to be optimized).
            5. **Problem Type**: Determine whether the model is a MILP (Mixed Integer Linear Programming) problem or an NLP (Nonlinear Programming) problem, and select one of the two values. 
            
            Note:
                1. List all variables, including those introduced for linearizing absolute differences, such as Δ⁺, Δ⁻ (if such variables exist, list them, otherwise leave them out), and generate the corresponding linearization constraints.  For each original sentence, scan and check if keywords like "change," "difference," "increment," "decrement," "change amount," etc., are mentioned, and generate the corresponding linearization constraints.
                2. "not need" does not necessarily mean that there are no variables or constraints. If an object appears in sentences such as "does not need to increase" or "will not increase", it may be necessary to consider the situation where the variable will decrease.
                3. Don’t ignore sentences starting with "In addition", "In addition to this", "By the way", etc., which may also contain information such as constraints or variables.
                4. If "all the sub-quotas" or "all types" are mentioned, then every category of situation must be considered.
            **Output** as follows:  
                  1.Sentence_Scanning
                    sentence 1:<sentence 1> -> <Constraint Scanning result description OR 'No constraints'>,
                    sentence 2:<sentence 2> -> <Constraint Scanning result description OR 'No constraints'>,
                    ...
                  2.Variables_List
                    Variable 1:...,
                    Variable 2:...,
                    ...
                  3.Constraint_Table
                    ["<Constraint 1 name>","<Mathematical expressions 1>","sentence numbers:<sentence numbers>"],
                    ["<Constraint 2 name>","<Mathematical expressions 2>","sentence numbers:<sentence numbers>"],
                    ...
                  4.Objective
                    <Objective sentence> and <Mathematical expressions>,
                    ...
                  5.Problem_Type
                    point <'MILP' OR 'NLP'>, and give description...
            '''
        )
        self.messages.append({"role": "system", "content": self.system_msg})

    async def generate(self, user_question, ):
        msgs = [
            {"role": "user", "content": (
                f'''
                    Here is the problem description:
                    ________________________________________
                    {user_question}
                    Output the lists(Sentence_Scanning, Variables_List, Constraint_Table,Objective, Problem_Type) mentioned above.'''
            )}
        ]
        self.messages.extend(msgs)
        return await self._query()

    async def get_and_change_format_output(self, origin_reply):
        self.messages.append({"role": "assistant", "content": (
            f"{origin_reply}"
        )})
        msgs = [
            {"role": "user", "content": (
                '''
                Now, please convert all your analysis results into the following JSON object format (no additional text):  
                For origin sentence, just write the first few words and "...".
                ```json
                {
                  "Sentence_Scanning": [
                    ["1","<origin sentence 1...>","<Constraint Scanning result description OR 'No constraints'>"],
                    ["2","<origin sentence 2...>","<Constraint Scanning result description OR 'No constraints'>"],
                    ...
                  ],
                  "Variables_List": [
                    {
                      "symbol":     "<chosen mathematical symbol>",
                      "Meaning":    "<parameter definition>",
                      "Type":       "<BINARY / integer / continuous type>",
                      "Range ":     "<Range of Values>"
                    },
                    ...
                  ],
                  "Constraint_Table":[
                    ["<Constraint 1 name>","<Mathematical expressions 1>","sentence numbers:<sentence numbers>"],
                    ["<Constraint 2 name>","<Mathematical expressions 2>","sentence numbers:<sentence numbers>"],
                    ...
                  ],
                  "Objective": {
                      "Objective_sentence":        "<Objective sentence>",
                      "Mathematical_expressions":  "<Mathematical expressions>"
                  },
                  "Problem_Type": "<'MILP' OR 'NLP'>"
                }```
            '''
            )}
        ]
        self.messages.extend(msgs)
        return await self._query()

    async def integrate_with_file(self, entry):
        """
        Load questions and answers from an existing JSON file, generate background, constraints, objective for each entry,
        merge them under respective keys, and write to a new JSON file.
        """
        question = entry.get('question', '')
        # Generate background, constraints, and objective via LLM
        try:
            origin_reply = await self.generate(question)
            origin_reply_format = await self.get_and_change_format_output(origin_reply)
            origin_reply_format_json = re.sub(r'```json|```', '', origin_reply_format).strip()
            targets_data = json.loads(origin_reply_format_json)
            # Merge the output with the existing entry
            entry['Sentence_Scanning'] = targets_data.get('Sentence_Scanning', [])
            entry['Variables_List'] = targets_data.get('Variables_List', [])
            entry['Constraint_Table'] = targets_data.get('Constraint_Table', [])
            entry['Objective'] = targets_data.get('Objective', {})
            entry['Problem_Type'] = targets_data.get('Problem_Type', "")
            print(f"Success ExtractFactorsFromLongTextAgent For Case {entry['index']}!")

        except Exception as e:
            # In case of error, log and continue
            print(f"Error generating Factors for Entry {entry['index']}: {e}")
        return entry


class ExtractParameterFromLongTextAgent(BaseAgent):
    def __init__(self, client, model_name="o3-mini", temperature=0.2):
        super().__init__(client, model_name, temperature)
        self.system_msg = (
            '''
            You are an expert in mathematical modeling and a professor of optimization at a top university. We will describe an optimization problem for you. Regarding this combinatorial optimization problem, please complete the following tasks:
            
            1. **Sentence Scanning**: Start by providing the original sentence number and content, and then scan sentence by sentence. EITHER extract it into one or more constraints information OR  just mark it as "No Values".
            2. **Extract Parameters**: Extract all parameters from the following paragraphs and tables, making sure that no elements in any sentence are omitted. Specifically, you need to give a **parameter list**, provide the names of all parameters, and must indicate the parameter type (integer/float/list/tuple) and give specific values.
            
            Note 1. The "Value" of the list/tuple type are defined by using the python format, and should not be string. Example, a list can use ["S", "V"] or a tuple type can use {"A": 450, "B": 400, "C": 300} and so on.
            Note 2. If the problem description contains **table** data (usually in markdown format), please strictly convert the table data into the form of a list or tuple in the python language. You must strictly refer to the data I provide and do not make up your own data. In the end, you must also extract the table data and name it Table_1_XXX, Table_2_XXX, and so on.
            Note 3. The step you are processing now is only used to find parameters with specific values, and you do not need to consider decision variables or other constraints!
            
            **Output** as follows:  
            1.Sentence Scanning Result
                sentence 1:<sentence 1> -> <Constraint Scanning result description OR 'No Values'>,
                sentence 2:<sentence 2> -> <Constraint Scanning result description OR 'No Values'>,
                ...
            2.Table Scanning Result
                table 1:<table_1_name> -> <Parameter Values(list/tuple)>,
                table 2:<table_2_name> -> <Parameter Values(list/tuple)>,
                ...
            '''
        )
        self.messages.append({"role": "system", "content": self.system_msg})

    async def generate(self, user_question, ):
        msgs = [
            {"role": "user", "content": (
                f'''
                    Here is the problem description:
                    ________________________________________
                    {user_question}
                    Output the result mentioned above.'''
            )}
        ]
        self.messages.extend(msgs)
        return await self._query()

    async def get_and_change_format_output(self, origin_reply):
        self.messages.append({"role": "assistant", "content": (
            f"{origin_reply}"
        )})
        msgs = [
            {"role": "user", "content": (
                '''
                Now, please convert all your analysis results(Sentence Parameters and Table Parameters) into **Parameters List**.
                
                Please adhere **strictly** to the following rules when generating the JSON field **"Value"**:
                1. Output must be **valid JSON**:  
                • All keys and string values in double quotes.  
                • No Python tuple syntax `(a, b)`.  
                • No objects with numeric or tuple keys.
                2.Value must follows these rules:  
                • If the original key is a **string**, keep the object structure.
                • If the original key is an **integer** (`0,1,2,…`), output a **one‐dimensional** array. Element at index i corresponds to the value for key i.  
                • If the original key is an **integer pair** `[i,j]`, output a **two‐dimensional square matrix**:  
                 
                 Now, use the following JSON object format (no additional text):  
                ```json
                {
                    "Parameters_List": [
                        {
                            "Name": "<Name of parameter1>",
                            "Type": "<integer/float/list/tuple>",
                            "Value": <Parameter Values, not string>,
                        },
                        ...
                    ]
                }
                '''
            )}
        ]
        self.messages.extend(msgs)
        return await self._query()

    async def integrate_with_file(self, entry):
        """
        Load questions and answers from an existing JSON file, generate background, constraints, objective for each entry,
        merge them under respective keys, and write to a new JSON file.
        """
        question = entry.get('question', '')
        # Generate background, constraints, and objective via LLM
        try:
            origin_reply = await self.generate(question)
            origin_reply_format = await self.get_and_change_format_output(origin_reply)
            origin_reply_format_json = re.sub(r'```json|```', '', origin_reply_format).strip()

            # origin_reply_format_json_fixed = quote_value_section(origin_reply_format_json)
            # origin_reply_format_json_fixed_fixed = fix_value_section(origin_reply_format_json_fixed)
            # targets_data = json.loads(origin_reply_format_json_fixed_fixed)
            targets_data = json.loads(origin_reply_format_json)

            # Merge the output with the existing entry
            entry['Parameters_List'] = targets_data.get('Parameters_List', [])
            print(f"Success ExtractParameter For Case {entry['index']}!")
        except Exception as e:
            # In case of error, log and continue
            print(f"Error generating Parameters for Entry {entry['index']}: {e}")
        return entry


# ---------------- 2.2.1 Modeling Agent  --------------------------------------------
class ModelingAgent(BaseAgent):
        def __init__(self, client, model_name="o3-mini", temperature=0.2):
            super().__init__(client, model_name, temperature)
            self.system_msg = (
                '''
                You are an operations research modeling expert. 
                Convert the user's optimization problem (natural language) into a precise mathematical model using linear programming expressions. 
                Focus on correct variable definitions, objective function, and constraints, without additional explanations.
                If you need to use the Big M method to control binary variables for linearization,  please note after this constraint: <When writing code, please use the function: model.addGenConstrIndicator(....).>
                e.g., f_A ≥ 900 implies a binary y_A, and you can use
                 \"model.addGenConstrIndicator(y_A,1,f_A >= 900)\",
                 \"model.addGenConstrIndicator(y_A,0,f_A <= 899)\"
                '''
            )
            self.messages.append({"role": "system", "content": self.system_msg})

        async def generate(self, entry: dict) -> str:
            # entry 包含 question, variables, background, constraints, objective
            context = json.dumps({
                'variables': entry.get('Variables_List', []),
                'constraints': entry.get('Constraint_Table', []),
                'objective': entry.get('Objective', {}),
                'problem_type':entry.get('Problem_Type', ''),
            }, ensure_ascii=False)
            self.messages.append({"role": "user", "content": (
                f"Problem: {entry['question']}\n"
                f"Context: {context}\n"
                f"Convert this into a precise model (variables, objective, constraints)."
            )})
            return await self._query()


# ---------------- 2.2.2 Auxiliary Model Agent  --------------------------------------------
class AuxiliaryModelAgent(BaseAgent):
    def __init__(self, client, model_name="o3-mini", temperature=0.2):
        super().__init__(client, model_name, temperature)
        self.system_msg = (
                """
            You are a leading mathematical modeling expert and optimization professor at a top university.
            Your task is to review a generated mathematical model and suggest encoding improvements 
            to ensure the subsequent coding agent can implement it without errors.

            Input:
            - The original problem entry, including question, variables, background, constraints, objective.
            - The  mathematical model coding advice for the subsequent Coding Agent .

            Requirements:
            1. Identify any function expressions that require auxiliary substitution variables, and use "model.Params.NonConvex = 2"
                e.g., $X^2$ requires $Y = X^2$, and propose the corresponding Gurobi "model.addGenConstrPow(X,Y,2)" statements.
                e.g., $log2(X)$ requires $Y = log2(X)$, and propose the corresponding Gurobi "model.addGenConstrLogA(X,Y,2)" statements.
            2. Gurobi cannot solve for variables in the denominator, and eliminate all denominator variables through variable substitution.
                e.g., if the variable involves 1/X1, you need to set an auxiliary substitution variable Y, 
                and the constraint must be "model.addConstr(X * Y == 1)" instead of "model.addConstr(Y == 1 / X)".
                e.g., do not use function:model.addGenConstrMul(X, L, XL),just use function:model.addConstr(X * L == XL)
            3. Detect any indicator-variable scenarios. 
                e.g., f_A ≥ 900 implies a binary y_A, and propose the necessary 
                "model.addGenConstrIndicator(y_A,1,f_A >= 900)",
                "model.addGenConstrIndicator(y_A,0,f_A <= 899)" constraints for both the 1 and 0 cases.
            4. If the objective does not involve these newly introduced variables, state "no need to modify".

            Output ONLY a JSON object with the following structure (no extra text):
            ```json
            {
              "math_model_advice": [
                {
                  "variables": [
                    "<sentences proposing new auxiliary variables>",
                    ...
                  ],
                  "constraints": [
                    "<sentences proposing new constraints>",
                    ...
                  ],
                  "objective": "<modified objective or 'no need to modify'>"
                }
              ]
            }
            ```
            """
        )
        self.messages.append({"role": "system", "content": self.system_msg})

    async def generate(self, entry: dict, math_model: str) -> str:
        # Prepare context
        context = json.dumps({
            'variables': entry.get('Variables_List', []),
            'constraints': entry.get('Constraint_Table', []),
            'objective': entry.get('Objective', {}),
            'problem_type': entry.get('Problem_Type', ''),
        }, ensure_ascii=False)

        # Compose user messages
        msgs = [
            {"role": "user", "content": (
                f"Problem: {entry['question']}\n"
                f"Context: {context}\n"
                f"Math model:\n{math_model}\n"
                "Please review and suggest encoding improvements as specified above."
            )}
        ]
        self.messages.extend(msgs)
        return await self._query()

    async def integrate_model(self, entry: dict, math_model: str) -> dict:
        advice_str = await self.generate(entry, math_model)
        advice_str = re.sub(r'```json|```', '', advice_str).strip()
        try:
            advice = json.loads(advice_str)
        except json.JSONDecodeError:
            advice = {'math_model_advice': []}
        return {
            'math_model': math_model,
            'math_model_advice': advice.get('math_model_advice', [])
        }

# ---------------- 2.2.3 Coding Agent  --------------------------------------------
class CodingAgent(BaseAgent):
    def __init__(self, client, model_name="o3-mini", problem_type="MILP", temperature=0.2):
        super().__init__(client, model_name, temperature)
        if problem_type == "NLP":
            system_msg = (
                "You are a Python program expert in the field of operations research and optimization, with proficiency in Gurobi Python coding. "
                "Given a user’s NLP(Nonlinear Programming) optimization problem, its identified variables, background, constraints, objective, and the validated mathematical model, "
                "generate complete, executable Gurobi code that follows this exact structure:\n"
                "   1. Import Gurobi and any other necessary packages.\n"
                "   2. Define all parameter matrices and data inputs.\n"
                "   3. Create decision variables.\n"
                "   4. Create any auxiliary substitution or indicator variables in coding advice"
                "       (The values of these auxiliary variables should range from"
                "        negative infinity to positive infinity, lb=-GRB.INFINITY, ub=GRB.INFINITY).\n"
                "   5. Set up the objective function.\n"
                "   6. Add all constraints (including gen‐constr and indicator constraints).\n"
                "   7. Solve the model and print results."
                "ATTENTION 1: You must add an extra statement at the end of the code to output the answer to the question,"
                "   and following the following format:"
                "   \"print(f\"FinalAnswer=【{the_question_answer}】\")\" "
                "   FinalAnswer has only one value, which is the value of the question, may or may not be the objective function.\n"
                "ATTENTION 2:\n"
                "   1. Identify any function expressions that require auxiliary substitution variables, and use \"model.Params.NonConvex = 2\" if needed.\n"
                "       e.g., $X^2$ requires $Y = X^2$, and propose the corresponding Gurobi \"model.addGenConstrPow(X,Y,2)\" statements. "
                "       Pay attention to the order of X and Y, don't reverse it."
                "       e.g., $log2(X)$ requires $Y = log2(X)$, and propose the corresponding Gurobi \"model.addGenConstrLogA(X,Y,2)\" statements. "
                "       Pay attention to the order of X and Y, don't reverse it."
                "   2. Gurobi cannot solve for variables in the denominator, and eliminate all denominator variables through variable substitution.\n"
                "       e.g., if the variable involves 1/X1, you need to set an auxiliary substitution variable Y, "
                "       and the constraint must be \"model.addConstr(X * Y == 1)\" instead of \"model.addConstr(Y == 1 / X)\"."
                "       e.g., do not use function:model.addGenConstrMul(X, L, XL),just use function:model.addConstr(X * L == XL)\n"
                "   3. If you find indicator-variable scenarios (for example, a variable needs to use different functions in different situations),"
                "        DO NOT use big-M for linearization, you need to use the \"addGenConstrIndicator\" function.\n"
                "       e.g., f_A ≥ 900 implies a binary y_A, and propose the necessary\n"
                "       \"model.addGenConstrIndicator(y_A,1,f_A >= 900)\","
                "       \"model.addGenConstrIndicator(y_A,0,f_A <= 899)\" constraints for both the 1 and 0 cases.\n"
                "Output only a fenced Python code block:\n"
                "```python\n"
                "{code}\n"
                "```"
            )
        else:
            system_msg = (
                "You are a Python program expert in the field of operations research and optimization, with proficiency in Gurobi Python coding. "
                "Given a user’s MILP(Mixed Integer Linear Programming) optimization problem, its identified variables, constraints, objective, and the validated mathematical model, "
                "generate complete, executable Gurobi code that follows this exact structure:\n"
                "   1. Import Gurobi and any other necessary packages.\n"
                "   2. Define all parameter matrices and data inputs.\n"
                "   3. Create decision variables.\n"
                "   5. Set up the objective function.\n"
                "   6. Add all constraints (DO NOT forget indicator constraints,if exist).\n"
                "   7. Solve the model and print results."
                "ATTENTION 1: You must add an extra statement at the end of the code to output the answer to the question,"
                "   and following the following format:"
                "   \"print(f\"FinalAnswer=【{the_question_answer}】\")\" "
                "   FinalAnswer has only one value, which is the value of the question, may or may not be the objective function.\n"
                "ATTENTION 2:\n"
                "   If you find indicator-variable scenarios (for example, a variable needs to use different functions in different situations),"
                "        DO NOT use big-M for linearization, you need to use the \"addGenConstrIndicator\" function.\n"
                "       e.g., f_A ≥ 900 implies a binary y_A, and propose the necessary\n"
                "       \"model.addGenConstrIndicator(y_A,1,f_A >= 900)\","
                "       \"model.addGenConstrIndicator(y_A,0,f_A <= 899)\" constraints for both the 1 and 0 cases.\n"
                "Output only a fenced Python code block:\n"
                "```python\n"
                "{code}\n"
                "```"
            )
        self.system_msg = system_msg
        self.messages.append({"role": "system", "content": self.system_msg})

    async def generate(self, entry, math_model, analysis=None, gurobi_code=None, side_info=None):

        context = json.dumps({
            'variables': entry.get('Variables_List', []),
            'constraints': entry.get('Constraint_Table', []),
            'objective': entry.get('Objective', {}),
            'problem_type': entry.get('Problem_Type', ''),
        }, ensure_ascii=False)

        Parameters_List = entry.get('Parameters_List', [])
        # 如果第一次生成，则需要输入所有的信息
        if gurobi_code is None and analysis is None and side_info is None:
            user_query = (
                f"Now, use these information to solve the mathematical problem.\n "
                f"Problem: {entry['question']}\n"
                f"Context: {context}\n"
                f"You must strictly use the Value in the Parameters List. You cannot rewrite it or make up your own.\n"
                f"Parameters List: {Parameters_List}\n, You only need to use the **Parameters List** I provided here. There is no need to extract data from other CSV format or table format files."

            )
            self.messages.append({"role": "user", "content": user_query})

        self.messages.append({"role": "assistant", "content": (
                f"Math model and coding advice:\n{math_model}\n"
            )})

        user_query2 = ""

        # 代码写错了或者没有最优解才会有gurobi_code analysis
        if gurobi_code and analysis:
            user_query2 = user_query2 + f"However, in actual generation, some problems may be encountered.\n "
            error_info = (
                f"You generated a code with errors!!!\n"
                f"Current code (with errors):\n{gurobi_code}\n"
                f"Analysis for Error code:\n{analysis}\n"
            )
            user_query2 = user_query2 + error_info

        # 运行错误或者没有最优解才会有side_info
        if side_info is not None:
            user_query2 = user_query2 + f"However, in actual generation, some problems may be encountered.\n "
            user_query2 = user_query2 + side_info
        else:
            just_re_generate = (
                "Based on the above, write complete and reliable Python code using Gurobi to solve "
                "this operations research optimization problem. "
                "In the generated Python code's first line, you must declare at the beginning: import gurobipy as gp"
                "When creating gurobi model, you must use variable name \"model\": model = gp.Model(\"XXXXXXXX\")"
            )
            user_query2 = user_query2 + just_re_generate

        self.messages.append({"role": "user", "content": user_query2})
        return await self._query()

# ---------------- 2.2.4 Repair Agent  --------------------------------------------
class RepairAgent(BaseAgent):
    def __init__(self, client, model_name="o3-mini", temperature=0.2):
        super().__init__(client, model_name, temperature)
        self.system_msg = (
            "You are an expert in Gurobi code debugging."
            "Diagnose problems and identify areas for improvement based on"
            " user inquiries, mathematical models, and error messages during code execution."
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

# ---------------- Agent Classes END --------------------------------------------



# ---------------- 3. 具体流程：先处理数据，然后让agent处理 -------------------------

# ---------------- 3.1 async 生成增强数据 -----------------------------------------

async def async_extract_and_execute_python_code(
    text_content, entry, output_dir=None, attempt=None,
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
                filename = f"case_{entry['index']}_{attempt+1}.py"
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
                best_obj = extract_best_objective(stdout_str)
                # return True, (str(best_obj) if best_obj is not None else stdout_str)
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



# ---------------- 3.3 async 生成与修复循环 -----------------------------------------
async def async_generate_or_code_solver(coder,repairer,entry, math_model,
                                        max_attempts=3,side_info = None, opts=None):
    runs_path = opts.output_dir
    coding = coder
    repair = repairer
    gurobi_code = await coding.generate(entry, math_model)
    print("【Python Gurobi Code】:\n", gurobi_code)
    attempt = 0
    while attempt < max_attempts:
        print(f"\n第 {attempt + 1} 次尝试，开始执行代码...\n")
        success, result = await async_extract_and_execute_python_code(gurobi_code,entry,runs_path,attempt)
        if success:
            return True, result, gurobi_code
        # 修复专家关闭
        # else:
        #     return False, None, gurobi_code
        # 修复专家启动
        print(f"\n第 {attempt + 1} 次尝试失败，请求 LLM 修复代码...\n")
        advise = await repair.generate(entry, math_model, result, gurobi_code)
        new_gurobi_code = await coding.generate(entry, math_model,analysis=advise,gurobi_code=gurobi_code, side_info=side_info)
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

    # 3) 调用写代码以及修复专家，生成并调试 Gurobi 代码
    coder = CodingAgent(async_openai, model_name=model_name, problem_type=problem_type)
    repair = RepairAgent(async_openai, model_name=model_name)

    # 4) 若成功或重试逻辑 unchanged...
    success, result, _ = await async_generate_or_code_solver(coder,repair, entry, math_model,
                                                             max_attempts=max_attempts,side_info=None, opts=opts)
    print(f'Stage result: {success}, {result}')
    if result is None:
        result = [None]
    if success:
        if is_number_string(str(result[0])):
            # 得到最优解
            return True, result
        else:
            # 没有最优解
            print('!![Run no available solution warning]!!')
            side_info = (
            "The model code still reports errors after multiple debugging attempts. Please carefully check if "
            "there are errors in the mathematical model. After checking, please rebuild the Gurobi Python code. "
            "Output in the format \n```python\n{code}\n```, without code explanations."
            )
            success, result, _ = await async_generate_or_code_solver(coder,repair, entry, math_model,
                                                                      max_attempts=3,side_info=side_info,
                                                                     opts=opts)
            return success, result
    else:
        # Run no success
        print('!![Run no success]!!')
        side_info = (
            "The model code still reports errors after multiple debugging attempts. Please carefully check if "
            "there are errors in the mathematical model. After checking, please rebuild the Gurobi Python code. "
            "Output in the format \n```python\n{code}\n```, without code explanations."
        )
        success, result, _ = await async_generate_or_code_solver(coder,repair, entry, math_model,
                                                                 max_attempts=3,side_info=side_info,
                                                                 opts=opts)
    return success, result


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

    return is_solve_success, result


# ----------------- 主流程与并发执行 -----------------------------------------
async def process_single_case(i, entry, args):
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
    # 执行代码
    if args.agent:
        ok, res = await async_NExT_OR_Agent(entry, args)
    else:
        ok, res = await async_gpt_code_agent_simple(entry, args)
    if isinstance(res, list):
        for res_i in res:
            if len(str(res_i))>20:
                res = None
                break
    elif isinstance(res, str):
        if len(res) > 20:
            res = None
    print("res=",res)

    # 记录执行结果
    if ok:
        print(f"成功执行代码，最优解值: {res}")
        result_data["execution"]["status"] = "success"
        result_data["execution"]["result"] = res
    else:
        print("执行代码失败。")
        result_data["execution"]["status"] = "failed"
        result_data["execution"]["result"] = res if 'res' in locals() else None

    pass_flag, correct_flag = eval_model_result(ok, res, ans)

    result_data["evaluation"] = {
        "pass_flag": pass_flag,
        "correct_flag": correct_flag,
        "run_result": res,
        "ground_truth": ans

    }

    print(f"Result: solve={ok}, value={res}, ground_truth={ans}")
    print(f'[Final] {i}run pass: {pass_flag}, solve correct: {correct_flag}')
    print(' ')

    # 保存结果到文件
    filename = f"case_{entry['index']}.txt"
    temp_file_path=os.path.join(output_dir, filename)
    with open(temp_file_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    return pass_flag, correct_flag, i

async def process_data_case(i, entry, args):
    print(f"============ Case {i} For Data Process ============")
    var_agent = ExtractFactorsFromLongTextAgent(async_openai, model_name=args.model)

    if args.multipara:
        para_agent = ExtractMultiParameterFromLongTextAgent(async_openai, model_name=args.model)
    else:
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
    os.makedirs(runs_ALL_path+"//NORA", exist_ok=True)
    opts.output_dir = runs_ALL_path

    base, ext = os.path.splitext(opts.dataset_name)
    new_v_ext = "o4mini_simple"
    opts.nora_file_input_path = f"{opts.output_dir}//NORA//{base}_NORA_{new_v_ext}.json"

    await main_process_data(opts)
    await main(opts)

async def main_process_data(opts=None):
    if opts is None:
        opts = await get_args()

    if opts.agent:
        opts.data_path = os.path.join("data/origin_data", f"{opts.dataset_name}"+".json")


    base, ext = os.path.splitext(opts.dataset_name)
    new_v_ext = "o4mini_simple"
    # 构建新的路径
    if opts.output_dir is None:
        nora_path = os.path.join("NORA", f"{os.path.basename(base)}_NORA_{new_v_ext}.json")
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

    # results = await asyncio.gather(*tasks)

    # # 创建一个 tqdm 实例
    # progress = tqdm(total=len(tasks), desc="Processing")
    #
    # # 给每个 coro 包一层
    # wrapped = [
    #     asyncio.create_task(_wrapper(task, progress))
    #     for task in tasks
    # ]
    #
    # # 并发执行所有 wrapped 任务
    # results = await asyncio.gather(*wrapped)
    #
    # # 任务跑完后关闭进度条
    # progress.close()


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

        fails[str(i)] =  full_entry
        # 推动进度条，并更新右侧的 postfix
        pbar.update(1)
        # 如果你还要打印每个 case 的细节，可以用 tqdm.write() 保证它们
        # 打印在进度条上方，而不会破坏进度条本身的位置：
        # tqdm.write(f"Case {idx:>3}: pass={p}, correct={c}")
    pbar.close()


    # fails = {str(i):full_entry for full_entry, i in results if full_entry}

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
    # opts.nora_file_input_path = "runs_ALL/【20250627_011339】_【optmath_bench_LP】_【o4-mini-2025-04-16-high】/NORA/optmath_bench_LP_NORA_o4mini_simple.json"

    if opts.nora_file_input_path is None:
        if opts.agent:
            new_v_ext = "o4mini"
            opts.data_path = "NORA/"+opts.dataset_name+f"_NORA_{new_v_ext}.json"
        else:
            opts.data_path = os.path.join("data/NExT_datasets", f"{opts.dataset_name}" + ".json")
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
        # w/o 逐句参数匹配机制
        # entry["Parameters_List"]=[]
        # w/o 长文本要素匹配机制
        entry["Sentence_Scanning"] = []
        entry["Variables_List"] = []
        entry["Constraint_Table"] = []
        entry["Objective"] = []
        # w/o 非线性识别机制 & 辅助变量模型
        # entry["Problem_Type"] = "MILP"
        task = process_single_case(i, entry, opts)
        tasks.append(task)

    # results = await asyncio.gather(*tasks)


    # # 创建一个 tqdm 实例
    # progress = tqdm(total=len(tasks), desc="Processing")
    #
    # # 给每个 coro 包一层
    # wrapped = [
    #     asyncio.create_task(_wrapper(task, progress))
    #     for task in tasks
    # ]
    #
    # # 并发执行所有 wrapped 任务
    # results = await asyncio.gather(*wrapped)
    #
    # # 任务跑完后关闭进度条
    # progress.close()


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
        ncols=100,        # 可选：固定宽度
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



    # pass_count = sum(1 for p, _, _ in results if p)
    # correct_count = sum(1 for _, c, _ in results if c)
    # fails = [i for p, c, i in results if not (p and c)]

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
    opts.output_dir = os.path.join("runs_single", f"【{timestamp}】_【{opts.dataset_name}】_【{opts.model}】_【{opts.singe_test}】")
    os.makedirs(opts.output_dir, exist_ok=True)
    nora_path = os.path.join(opts.output_dir, f"{opts.dataset_name}_{str(opts.singe_test)}_NORA.json")

    task_list = []
    if opts.singe_test is not None and opts.singe_test !="":
        task_list = [str(x.strip()) for x in opts.singe_test.split(",")]
    print("task_list=", task_list)

    if opts.test_nora is not None:
        print("opts.test_nora is not None, and use exist file.")
        exist_path = os.path.join("runs_single", opts.test_nora, f"{opts.dataset_name}_{str(opts.singe_test)}_NORA.json")
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
    # asyncio.run(main_process_data())
    asyncio.run(main_all())