import json

from agents.base_agents import BaseAgent


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
