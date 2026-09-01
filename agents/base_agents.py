# 所有的初始agent定义
import re


class BaseAgent:
    def __init__(self, client, model_name="gpt-5", temperature=0.2):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.messages = []
        self.total_tokens = 0  # 初始化token计数器
        self.returned_models = []

    async def _query(self):
        resp = await self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
        )
        returned_model = getattr(resp, "model", None)
        # Record that a provider completion was observed before enforcing the
        # exact-model protocol. This prevents one-shot experiments from
        # accidentally retrying a mismatched response as if no call occurred.
        self.returned_models.append(returned_model)
        if returned_model != self.model_name:
            raise RuntimeError(
                "MODEL_ID_MISMATCH: requested "
                f"{self.model_name!r}, returned {returned_model!r}"
            )
        # 累加本次调用的token
        if resp.usage:
            self.total_tokens += resp.usage.total_tokens

        content = resp.choices[0].message.content
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        self.messages.append({"role": "assistant", "content": content})
        return content

# 一次调用模型：直接建模，输出GUROBI代码
class Simple_agent(BaseAgent):
    def __init__(self, client, model_name="gpt-5", temperature=0.2):
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
