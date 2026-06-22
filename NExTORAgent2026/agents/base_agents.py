# 所有的初始agent定义
import re


class BaseAgent:
    def __init__(self, client, model_name="gpt-5", temperature=0.2):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.messages = []
        self.total_tokens = 0  # 初始化token计数器

    async def _query(self):
        resp = await self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
        )
        # 累加本次调用的token
        if resp.usage:
            self.total_tokens += resp.usage.total_tokens

        content = resp.choices[0].message.content
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        self.messages.append({"role": "assistant", "content": content})
        return content

# 现在，大模型在运筹优化领域，还有什么比较前沿的领域比较有潜力开发一下？
# 大模型建模侧重在建模，而不是信息收集，

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


#
# class BaseAgent:
#     def __init__(self, client, model_name="o3-mini", temperature=0.2):
#         self.client = client
#         self.model_name = model_name
#         self.temperature = temperature
#         self.messages = []
#         self.total_tokens = 0  # 初始化token计数器
#
#     async def _query(self):
#         resp = await self.client.chat.completions.create(
#             model=self.model_name,
#             messages=self.messages,
#             temperature=self.temperature
#         )
#         # 累加本次调用的token
#         if resp.usage:
#             self.total_tokens += resp.usage.total_tokens
#
#         content = resp.choices[0].message.content
#         self.messages.append({"role": "assistant", "content": content})
#         return content
#
#
# # 一次调用模型：直接建模，输出GUROBI代码
# class Talk_agent(BaseAgent):
#     def __init__(self, client, model_name="o3-mini", temperature=0.2):
#         super().__init__(client, model_name, temperature)
#         self.system_msg = (
#             '''
#            You are an operations optimization expert.
#            Please answer the problem provided by the user.
#             '''
#         )
#         self.messages.append({"role": "system", "content": self.system_msg})
#
#     async def generate(self, ask: dict) -> str:
#         self.messages.extend([{"role": "user", "content": f"{ask}"}])
#         return await self._query()