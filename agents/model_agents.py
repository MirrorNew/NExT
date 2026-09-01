from agents.base_agents import BaseAgent
import json, re

class ModelingAgent(BaseAgent):
    def __init__(self, client, model_name="o3-mini", temperature=0.2):
        super().__init__(client, model_name, temperature)
        self.system_msg = (
            '''
            You are an operations research modeling expert. 
            Convert the user's optimization problem (natural language) into a precise mathematical model using the required linear or nonlinear optimization expressions.
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
            'problem_type': entry.get('Problem_Type', ''),
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
        2. Modern Gurobi versions support nonlinear expressions, including supported variable-denominator expressions, through native nonlinear interfaces.
            Before encoding 1/X1, preserve the required domain condition X1 != 0. You may use a supported native nonlinear expression or introduce an auxiliary reciprocal variable Y
            with "model.addConstr(X1 * Y == 1)" when an explicit decomposition is needed.
            e.g., DO NOT not use function:model.addGenConstrMul(X, L, XL),just use function:model.addConstr(X * L == XL)
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
