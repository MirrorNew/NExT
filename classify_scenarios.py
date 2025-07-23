import json
import openai
import os
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from data.make_file_format_as_json import make_file_format_as_json

# 设置样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 16+4
# plt.rcParams['axes.titlesize'] = 24+4
# plt.rcParams['axes.labelsize'] = 20+4
# plt.rcParams['xtick.labelsize'] = 18
# plt.rcParams['ytick.labelsize'] = 18+4
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'




openai_api_data = dict(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

# API 客户端 Shared async clients
async_openai = openai.AsyncOpenAI(
    api_key=openai_api_data['api_key'],
    base_url=openai_api_data['base_url'] or None
)

class BaseAgent:
    def __init__(self, client, model_name="o4-mini", temperature=0.2):
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


class C1_Agent(BaseAgent):
    def __init__(self, client, model_name="o3-mini", temperature=0.2):
        super().__init__(client, model_name, temperature)
        self.system_msg = (
            '''
                You are an expert in classifying questions into application domains. Your task is to classify a given question into **one and only one** of the following categories. The output must be a **single word**, chosen as follows:
                
                - If the question clearly fits one of the predefined categories below, return the category name **exactly as listed**.
                - If the question does not fit any of the categories, return a **single-word label** that best describes the question's actual domain or topic.
                
                Predefined categories:
                - Agriculture
                - Energy
                - Health
                - Retail
                - Environment
                - Education
                - Financial Services
                - Transportation
                - Public Utilities
                - Manufacturing
                - Software
                - Construction
                - Legal
                - Customer Service
                - Entertainment
                
                Your output must be a **single word** — either a category name from the list above, or a one-word domain name you believe is more appropriate. Do not include any explanation or punctuation.
            '''
        )
        self.messages.append({"role": "system", "content": self.system_msg})

    async def generate(self, entry: dict) -> str:
        self.messages.append({
            "role": "user",
            "content": f"Question: {entry['question']}\nPlease output only one word representing the category.(e.g. 'Retail')."
        })
        return await self._query()




class C2_Agent(BaseAgent):
    def __init__(self, client, model_name="o3-mini", temperature=0.2):
        super().__init__(client, model_name, temperature)
        self.system_msg = (
            '''
            你是一个优化问题分类助手。请根据问题内容，仅从以下类型中选择最合适的一类作为输出结果，严格只输出一个类别：
            
            - IP （纯整数规划）
            - LP （纯线性规划）
            - MILP （混合整数线性规划）
            - QP （二次规划）
            - SOCP （二阶锥规划）
            - HighP （高次幂非线性规划，幂大于2）
            - FracP （带有非整数幂的非线性规划）
            - ELP （包含指数EXP或对数项LOG的非线性规划）
            
            你的回答必须只输出以上8类中的**一个英文缩写**，不允许解释或附加说明。
            '''
        )
        self.messages.append({"role": "system", "content": self.system_msg})

    async def generate(self, entry: dict) -> str:
        self.messages.append({
            "role": "user",
            "content": f"Problem: {entry['question']}\nPlease only return one of the above types (e.g., 'MILP')."
        })
        return await self._query()


class VN_Agent(BaseAgent):
    def __init__(self, client, model_name="o3-mini", temperature=0.2):
        super().__init__(client, model_name, temperature)
        self.system_msg = (
            '''
            你是一个优化问题分析助手。请根据问题要求，给出答案：仅给出数字：

            1. 读取代码，并分析代码之后，给出变量数与约束的数量
            注意

            你的回答必须只输出**变量的数量数字**，不允许解释或附加说明。
            '''
        )
        self.messages.append({"role": "system", "content": self.system_msg})

    async def generate(self, entry: dict) -> str:
        self.messages.append({
            "role": "user",
            "content": f"Problem: {entry['question']}\nPlease only return one of the above types (e.g., 'MILP')."
        })
        return await self._query()


# 定义一个函数来判断问题的类别
async def classify_question(i, question):
    c1_agent = C1_Agent(client=async_openai)
    c2_agent = C2_Agent(client=async_openai)
    # 异步调用C1和C2代理获取结果
    c1_classify_res = await c1_agent.generate({"question": question})
    c2_classify_res = await c2_agent.generate({"question": question})
    print(f"finish index:{i}")
    return c1_classify_res, c2_classify_res, i


async def main_classify_question():
    input_path1 = "data/origin_data/NExT_LP.json"
    input_path2 = "data/origin_data/NExT_NLP.json"
    print(f"input_path1={input_path1}")
    print(f"input_path2={input_path2}")

    with open(input_path1, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    with open(input_path2, 'r', encoding='utf-8') as f:
        dataset2 = json.load(f)

    dataset_all = {}
    dataset_all.update(dataset)
    for k, v in dataset2.items():
            new_key = str(int(k)+ 38)
            dataset_all[new_key] = v

    tasks = []
    for i, d in dataset_all.items():
        task = classify_question(i, d['question'])  # Assuming 'question' key contains the problem
        tasks.append(task)

    # 获取所有任务的结果
    results = await asyncio.gather(*tasks)

    # 初始化结果字典
    c1_results = {}
    c2_results = {}

    # 收集每个问题的分类结果
    for c1_result, c2_result, i in results:
        c1_results[i] = c1_result
        c2_results[i] = c2_result

    # 统计类别的频率
    c1_df = pd.DataFrame(list(c1_results.items()), columns=["Index", "C1_Category"])
    c2_df = pd.DataFrame(list(c2_results.items()), columns=["Index", "C2_Category"])

    # 统计分类的频率
    c1_frequency = c1_df["C1_Category"].value_counts(normalize=True) * 100
    c2_frequency = c2_df["C2_Category"].value_counts(normalize=True) * 100
    print(c1_frequency)
    print(c2_frequency)

    # 构造保存内容
    output_data = {
        "C1_Frequency": c1_frequency.to_dict(),
        "C2_Frequency": c2_frequency.to_dict()
    }
    # 写入 JSON 文件
    output_path = os.path.join("D:\\LLMProject\\NExTORAgent2025", "classification_summary.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"\n分类结果已保存至: {output_path}")

def draw():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # 使用相同的数据
    categories = ['IP', 'LP', 'MILP', 'QP', 'SOCP', 'HighP', 'FracP', 'ELP']
    datasets = ['NL4OPT', 'MAMO EasyLP', 'MAMO ComplexLP', 'OptMATH-Bench', 'NExT']

    data = {
        'NL4OPT': [26.94, 61.63, 11.43, 0.00, 0.00, 0.00, 0.00, 0.00],
        'MAMO EasyLP': [58.74, 0.61, 40.65, 0.00, 0.00, 0.00, 0.00, 0.00],
        'MAMO ComplexLP': [15.64, 62.08, 22.28, 0.00, 0.00, 0.00, 0.00, 0.00],
        'OptMATH-Bench': [11.92, 8.81, 67.87, 5.18, 6.22, 0.00, 0.00, 0.00],
        'NExT': [2.631579, 10.526316, 38.157895, 14.473684, 2.631579, 3.947368, 25.000000, 2.631579],
    }

    df = pd.DataFrame(data, index=categories)

    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制正方形热力块（带间隙，正方形单元）
    cell_size = 1.0
    gap = 0.1
    square_size = cell_size - gap

    n_rows, n_cols = len(datasets), len(categories)

    for i in range(n_rows):
        for j in range(n_cols):
            value = df.iloc[j, i]
            # normed_value = np.sqrt(value / 100)  # √归一化增强低值颜色
            sss = 1
            normed_value = np.log1p(sss * value) / np.log1p(sss * 100)
            color = plt.cm.Blues(normed_value)
            # 每个正方形的位置
            rect = plt.Rectangle(
                (j + gap / 2, i + gap / 2),
                square_size,
                square_size,
                facecolor=color,
                edgecolor='none'
            )
            ax.add_patch(rect)
            text_color = 'black' if value < 50 else 'white'
            # ax.text(j + 0.5, i + 0.5, f"{value:.2f}%", ha='center', va='center', color=text_color, fontsize=8)

    # 设置坐标轴刻度
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    # ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_xticklabels(categories, rotation=0, ha='center')
    ax.set_yticklabels(datasets)
    ax.invert_yaxis()

    # 去除边框线和坐标轴
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False)

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0, 100))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Percentage")

    ax.set_title("Problem Type Distribution Across Datasets")

    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

def draw2():
    import matplotlib.pyplot as plt
    import numpy as np

    # 原始数据
    data = {
        "Manufacturing": 32.89,
        "Energy": 13.16,
        "Transportation": 13.16,
        "Health": 7.89,
        "Financial Services": 7.89,
        "Retail": 6.58,
        "Agriculture": 5.26,
        "Construction": 5.26,
        "Public Utilities": 2.63,
        "Security": 1.32,
        "Entertainment": 1.32,
        "Education": 1.32,
        "Telecom": 1.32
    }

    labels = list(data.keys())
    values = np.array(list(data.values()))
    num_sectors = len(values)

    # 归一化：最短为 r_base + r_min_padding，最长为 r_max
    r_base = 1.0
    r_max = 5.0
    r_min_padding = 1.0
    normalized_values = values / values.max()
    r_scaled = r_base + r_min_padding + normalized_values * (r_max - r_base - r_min_padding)

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': 'polar'})
    cmap = plt.cm.Blues

    # 绘图
    theta = np.linspace(0, 2 * np.pi, num_sectors + 1)

    for i in range(num_sectors):
        theta_start = theta[i]
        theta_end = theta[i + 1]
        width = theta_end - theta_start
        radius = r_scaled[i]
        value = values[i]

        # 创建梯形扇形路径
        angles = np.linspace(theta_start, theta_end, 30)
        inner_arc = np.column_stack([angles, np.full_like(angles, r_base)])
        outer_arc = np.column_stack([angles[::-1], np.full_like(angles, radius)])
        arc = np.concatenate([inner_arc, outer_arc])

        # 颜色加亮处理：将最大值颜色限制在 80% colormap 强度
        color_strength = 0.2 + 0.7 * normalized_values[i]
        face_color = cmap(color_strength)

        ax.fill(arc[:, 0], arc[:, 1], color=face_color, edgecolor='white')

        # 文本位置：沿中轴线放置，不旋转
        mid_angle = (theta_start + theta_end) / 2
        text_radius = radius + 0.3
        ax.text(mid_angle, text_radius, f"{labels[i]}\n{value:.1f}%", ha='center', va='center', fontsize=8, rotation=0)

    # 添加颜色条
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    # norm = Normalize(vmin=min(values), vmax=max(values))
    # sm = ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    # cbar.set_label("Percentage")

    norm = Normalize(vmin=min(values), vmax=max(values))
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.6)  # 缩短颜色条
    cbar.set_label("Percentage")


    # 美化图形：只保留最外层圆圈
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.set_xticklabels([])

    # 美化
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title("Category Distribution (Normalized Radial Trapezoid Chart)", va='bottom')
    ax.set_ylim(0, r_max + 1)
    plt.tight_layout()
    plt.show()

def draw3():
    import matplotlib.pyplot as plt
    import numpy as np

    # # 数据集名称和对应的平均长度（中文算两个字符）
    # datasets = [
    #     'NExTOR(L)','NExTOR', 'OPTMATH', 'IndustryOR', 'OptiBench', 'ORllmAgent',
    #     'MAMO(H)', 'ComplexOR', 'NL4OPT', 'NLP4LP'
    # ]
    # avg_lens = [3594,3144, 2974, 1185, 686, 1220, 1724, 1273, 534, 536]


    # 使用相同的数据
    # categories = ['IP', 'LP', 'MILP', 'QP', 'SOCP', 'HighP', 'FracP', 'ELP']
    # datasets = ['NL4OPT', 'MAMO EasyLP', 'MAMO ComplexLP', 'OptMATH-Bench', 'NExT']
    #
    # data = {
    #     'NL4OPT': [26.94, 61.63, 11.43, 0.00, 0.00, 0.00, 0.00, 0.00],
    #     'MAMO EasyLP': [58.74, 0.61, 40.65, 0.00, 0.00, 0.00, 0.00, 0.00],
    #     'MAMO ComplexLP': [15.64, 62.08, 22.28, 0.00, 0.00, 0.00, 0.00, 0.00],
    #     'OptMATH-Bench': [11.92, 8.81, 67.87, 5.18, 6.22, 0.00, 0.00, 0.00],
    #     'NExT': [2.631579, 10.526316, 38.157895, 14.473684, 2.631579, 3.947368, 25.000000, 2.631579],
    # }

    datasets =['IP', 'LP', 'MILP', 'QP', 'SOCP', 'HighP', 'FracP', 'ELP']
    avg_lens = [2.631579, 10.526316, 38.157895, 14.473684, 2.631579, 3.947368, 25.000000, 2.631579]



    # 归一化颜色强度
    norm = plt.Normalize(min(avg_lens), max(avg_lens))
    colors = plt.cm.Blues(norm(avg_lens))

    # 绘图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(datasets, avg_lens, color=colors)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 50, f'{height}', ha='center', va='bottom')

    # 设置标题和标签
    plt.title('Comparison of Average Length per Dataset (Chinese counts as 2 chars)')
    plt.ylabel('Average Length')
    plt.xticks(rotation=0)

    # 仅保留横纵轴，去除右边和顶部边框
    ax = plt.gca()


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.yaxis.grid(True, linestyle='-', alpha=0.6)
    ax.xaxis.grid(False)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()

def draw4():
    input_path1 = "data/origin_data/NExT_LP.json"
    input_path2 = "data/origin_data/NExT_NLP.json"
    print(f"input_path1={input_path1}")
    print(f"input_path2={input_path2}")

    with open(input_path1, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    with open(input_path2, 'r', encoding='utf-8') as f:
        dataset2 = json.load(f)

    data = {}
    data.update(dataset)
    for k, v in dataset2.items():
            new_key = str(int(k)+ 38)
            data[new_key] = v

    # dataset_name = "NExT_LP.json"
    # # Path to the JSON file
    # json_file_path = f"data/NExT_datasets/{dataset_name}"
    #
    # with open(json_file_path, 'r', encoding='utf-8') as file:
    #     if dataset_name.endswith(".json"):
    #         # Read the JSON file
    #         data = json.load(file)
    #     else:
    #         data = make_file_format_as_json(dataset_name)
    # 收集问题长度（每个中文算两个字符）
    def count_length(text):
        return sum(2 if '\u4e00' <= ch <= '\u9fff' else 1 for ch in text)

    question_lengths = [count_length(item["question"]) for item in data.values() if "question" in item]


    # 直方图绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    min_length = min(question_lengths)
    max_length = max(question_lengths)
    # bin_edges = np.arange(min_length - min_length % 100, max_length + 100, 100)
    bin_edges = np.arange(min_length - min_length % 250, max_length + 250, 250)

    # 绘制直方图，获得频率值
    hist_vals, bins, patches = ax.hist(question_lengths, bins=bin_edges.tolist(), density=False,
                                       edgecolor='white', linewidth=0.5)
    # 添加百分比标签（每个柱子顶部）
    for patch, freq in zip(patches, hist_vals):
        height = patch.get_height()
        if height > 0:
            ax.text(patch.get_x() + patch.get_width() / 2, height,
                    f'{freq/76.0*100 :.2f}%',  # 转换为百分比
                    ha='center', va='bottom', fontsize=16)

    # 用频率值设置颜色深浅：越高越深蓝
    norm = plt.Normalize(min(hist_vals), max(hist_vals))
    for patch, freq in zip(patches, hist_vals):
        patch.set_facecolor(plt.cm.Blues(norm(freq)))

    # KDE 曲线
    x_values = np.linspace(min_length, max_length, 1000)
    kde = stats.gaussian_kde(question_lengths, bw_method='scott')
    density_curve = kde(x_values)
    density_curve_scaled = density_curve * len(question_lengths) * (bins[1] - bins[0])
    # ax.plot(x_values, density_curve, color=plt.cm.Blues(0.9), linewidth=2)
    ax.plot(x_values, density_curve_scaled, color='red', linewidth=2)



    # 轴标签与标题
    # ax.set_xlabel('Question Length (characters)')
    ax.set_ylabel('Frequency Distribution')
    ax.set_title('Question Length Distribution (Blue: darker = higher frequency)')

    # 统计信息
    # stats_text = (f"Total Questions: {len(question_lengths)}\n"
    #               f"Min Length: {min_length}\n"
    #               f"Max Length: {max_length}\n"
    #               f"Average Length: {sum(question_lengths) / len(question_lengths):.2f}\n"
    #               f"Median Length: {np.median(question_lengths):.2f}")
    # ax.text(0.73, 0.95, stats_text, transform=ax.transAxes, fontsize=16, verticalalignment='top',
    #         fontfamily='Times New Roman', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5))

    # 去除边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 网格线设置
    ax.yaxis.grid(True, linestyle='-', alpha=0.6)
    ax.xaxis.grid(False)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()

def draw5():


    # Data
    conditions = [
        'Only Prompt (o4-mini)',
        'w/o SbS Parameters Match',
        'w/o SbS Elements Match',
        'w/o Nonlinear Model Align',
        'w/o Errors Advice',
        'NORA'
    ]

    # Metrics
    nonlinear_AC = np.array([60.53, 81.58, 86.84, 65.79, 81.58, 92.11])
    nonlinear_PR = np.array([89.47, 92.11, 89.47, 94.73, 86.84, 100.00])

    linear_AC = np.array([44.74, 57.89, 52.63, 57.89, 60.53, 60.53])
    linear_PR = np.array([89.47, 97.37, 97.37, 100.00, 86.84, 100.00])

    avgAC = np.array([52.63, 69.73, 68.42, 61.84, 71.05, 76.31])
    avgPR = np.array([89.47, 94.74, 93.42, 97.37, 88.16, 100.00])

    # Delta relative to NORA
    delta_nonlin_AC = nonlinear_AC - nonlinear_AC[-1];
    delta_nonlin_AC[[0, -1]] = 0
    delta_nonlin_PR = nonlinear_PR - nonlinear_PR[-1];
    delta_nonlin_PR[[0, -1]] = 0
    delta_lin_AC = linear_AC - linear_AC[-1];
    delta_lin_AC[[0, -1]] = 0
    delta_lin_PR = linear_PR - linear_PR[-1];
    delta_lin_PR[[0, -1]] = 0
    delta_avgAC = avgAC - avgAC[-1];
    delta_avgAC[[0, -1]] = 0
    delta_avgPR = avgPR - avgPR[-1];
    delta_avgPR[[0, -1]] = 0

    # Colors
    colors = plt.cm.tab10.colors

    # Plot setup
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    height = 0.12
    y = np.arange(2)

    # Plot 1: Nonlinear
    for i, cond in enumerate(conditions):
        axes[0].barh(y + (i - 2.5) * height, [nonlinear_AC[i], nonlinear_PR[i]], height, color=colors[i])
    axes[0].plot(delta_nonlin_AC, y[0] + (np.arange(len(conditions)) - 2.5) * height, 'o-', color='black')
    axes[0].plot(delta_nonlin_PR, y[1] + (np.arange(len(conditions)) - 2.5) * height, '^--', color='black')
    axes[0].set_yticks(y);
    axes[0].set_yticklabels(['AC', 'PR']);
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Percentage (%)');
    axes[0].set_title('Nonlinear Tasks')

    # Plot 2: Linear
    for i in range(len(conditions)):
        axes[1].barh(y + (i - 2.5) * height, [linear_AC[i], linear_PR[i]], height, color=colors[i])
    axes[1].plot(delta_lin_AC, y[0] + (np.arange(len(conditions)) - 2.5) * height, 'o-', color='black')
    axes[1].plot(delta_lin_PR, y[1] + (np.arange(len(conditions)) - 2.5) * height, '^--', color='black')
    axes[1].set_yticks(y);
    axes[1].set_yticklabels(['AC', 'PR']);
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Percentage (%)');
    axes[1].set_title('Linear Tasks')

    # Plot 3: Overall Avg
    for i in range(len(conditions)):
        axes[2].barh(y + (i - 2.5) * height, [avgAC[i], avgPR[i]], height, color=colors[i])
    axes[2].plot(delta_avgAC, y[0] + (np.arange(len(conditions)) - 2.5) * height, 'o-', color='black')
    axes[2].plot(delta_avgPR, y[1] + (np.arange(len(conditions)) - 2.5) * height, '^--', color='black')
    axes[2].set_yticks(y);
    axes[2].set_yticklabels(['AvgAC', 'AvgPR']);
    axes[2].invert_yaxis()
    axes[2].set_xlabel('Percentage (%)');
    axes[2].set_title('Overall Average Metrics')

    # Global legend on right
    patches = [mpatches.Patch(color=colors[i], label=conditions[i]) for i in range(len(conditions))]
    line_AC = mlines.Line2D([], [], color='black', marker='o', label='Δ AC')
    line_PR = mlines.Line2D([], [], color='black', marker='^', linestyle='--', label='Δ PR')
    fig.legend(handles=patches + [line_AC, line_PR], loc='center right', bbox_to_anchor=(1.15, 0.5))

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


if __name__ == '__main__':
    # asyncio.run(main_classify_question())
    # draw()
    draw5()