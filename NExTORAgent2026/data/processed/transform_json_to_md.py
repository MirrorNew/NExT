import json
import os
import re


def convert_json_to_md_with_math(input_filename, output_filename):
    """
    将包含运筹学问题的JSON文件转换为Markdown文件。
    主要功能：
    1. Key 转为一级标题
    2. "question" 转为二级标题
    3. 自动识别并转换 LaTeX 公式格式：
       \( ... \) -> $ ... $
       \[ ... \] -> $$ ... $$
    """

    if not os.path.exists(input_filename):
        print(f"错误: 找不到文件 {input_filename}")
        return

    try:
        # 读取 JSON 文件
        with open(input_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        with open(output_filename, 'w', encoding='utf-8') as md_file:
            # 排序 Keys 保证顺序
            sorted_keys = sorted(data.keys())

            for prob_id in sorted_keys:
                item = data[prob_id]

                # 1. 一级标题: key
                md_file.write(f"# {prob_id}\n\n")

                # 2. 二级标题: question 及内容处理
                if "question" in item:
                    md_file.write("## question\n\n")

                    text = item["question"]

                    # --- 核心修改：公式格式替换 ---
                    # 替换行内公式 \( ... \) 为 $ ... $
                    # 注意：json.load 后，JSON中的 "\\" 已经被转义为 "\"
                    text = text.replace(r'\(', '$').replace(r'\)', '$')

                    # 替换块级公式 \[ ... \] 为 $$ ... $$
                    text = text.replace(r'\[', '$$').replace(r'\]', '$$')
                    # ---------------------------

                    md_file.write(text)
                    md_file.write("\n\n")

                # 3. 其他字段
                md_file.write("### Other Details\n")
                for key, value in item.items():
                    if key == "question":
                        continue
                    val_str = str(value) if value is not None else ""
                    # 为了防止其他字段里的 markdown 字符破坏格式，可以简单处理，也可以保留原样
                    md_file.write(f"- **{key}**: {val_str}\n")

                md_file.write("\n---\n\n")

        print(f"转换成功！\n输入文件: {input_filename}\n输出文件: {output_filename}")
        print("公式符号已完成替换：\\( -> $ 以及 \\[ -> $$")

    except json.JSONDecodeError:
        print("错误: JSON 文件格式不正确，请检查文件内容。")
    except Exception as e:
        print(f"发生未知错误: {e}")


def fill_answers_from_ground_truth(source_file, target_file):
    """
    读取 source_file (sample_LP_100_Chinese.md) 中的 ground_truth，
    并将其填充到 target_file (sample_LP_100_Chinese_A.md) 中对应的 wait_to_get 位置。
    """

    # 字典用于存储提取到的真值，格式: {'prob_001': '100', 'prob_002': '180000.5', ...}
    truth_map = {}

    # === 第一步：读取源文件，提取 ground_truth ===
    print(f"正在读取源文件: {source_file} ...")
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            current_prob_id = None

            for line in f:
                # 1. 识别一级标题 # prob_XXX
                prob_match = re.match(r'^#\s+(prob_\d+)', line)
                if prob_match:
                    current_prob_id = prob_match.group(1)
                    continue

                # 2. 在当前 prob_XXX 下寻找 ground_truth
                # 正则解释：匹配 "- **ground_truth**: " 后面的所有内容（去除首尾空格）
                if current_prob_id:
                    gt_match = re.search(r'-\s*\*\*ground_truth\*\*:\s*(.+)', line)
                    if gt_match:
                        # 提取值 (例如 "180000")
                        value = gt_match.group(1).strip()
                        truth_map[current_prob_id] = value
                        # print(f"提取到 {current_prob_id}: {value}") # 调试用

    except FileNotFoundError:
        print(f"错误: 找不到文件 {source_file}")
        return

    print(f"共提取到 {len(truth_map)} 个答案。正在处理目标文件...")

    # === 第二步：读取目标文件，替换 wait_to_get ===
    new_lines = []
    modified_count = 0

    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            current_prob_id = None

            for line in f:
                # 1. 追踪当前处理的是哪个 prob_XXX
                prob_match = re.match(r'^#\s+(prob_\d+)', line)
                if prob_match:
                    current_prob_id = prob_match.group(1)
                    new_lines.append(line)
                    continue

                # 2. 检查是否是需要替换的行
                # 逻辑：必须在某个 prob ID 下，且该 ID 在我们的字典里，且该行包含 "wait_to_get"
                if (current_prob_id and
                        current_prob_id in truth_map and
                        "**answer**" in line and
                        "wait_to_get" in line):

                    ground_truth_val = truth_map[current_prob_id]

                    # 使用正则进行精确替换，保留原有的缩进和格式
                    # 将 "wait_to_get" 替换为 ground_truth_val
                    new_line = re.sub(r'wait_to_get', ground_truth_val, line)

                    new_lines.append(new_line)
                    modified_count += 1
                else:
                    new_lines.append(line)

    except FileNotFoundError:
        print(f"错误: 找不到文件 {target_file}")
        return

    # === 第三步：写入结果回目标文件 ===
    with open(target_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"处理完成！")
    print(f"文件 {target_file} 已更新，共替换了 {modified_count} 处 wait_to_get。")



def extract_numbers(desc):
    # 使用正则表达式提取所有包含【数字】或【数字-数字】的项
    pattern = r'【([^】]+)】'
    matches = re.findall(pattern, desc)

    # 存储所有提取的数字
    numbers = set()

    for match in matches:
        # 处理逗号（中文或英文逗号）分隔的数字
        items = re.split(r'[，,]', match)

        for item in items:
            # 处理范围表达式（如1-3）
            if '-' in item:
                start, end = item.split('-')
                start, end = int(start), int(end)
                numbers.update(range(start, end + 1))
            else:
                numbers.add(int(item))

    # 打印并返回去重后的整数集合
    print(sorted(numbers))
    return sorted(numbers)

if __name__ == '__main__':

    # 示例调用
    desc = '''2024年底至2025年，人工智能领域经历了一场深刻的范式转移。如果说2023年至2024年是大型语言模型（LLM）通过扩大参数规模和训练数据量来追求“广度”的时期，那么2025年则标志着这一领域向“深度”的进军。随着OpenAI的o3系列、DeepSeek的R1系列以及Meta的Llama 4等模型的发布，行业焦点从单纯的语言流利度转向了复杂的逻辑推理、长程规划以及自我修正能力。这一转变的核心在于“系统2”（System 2）思维的引入。借鉴认知心理学家丹尼尔·卡尼曼的理论，传统的LLM主要模拟了人类的“系统1”——即快速、直觉、基于模式匹配的反应；而2025年的新一代模型则通过强化学习（RL）、思维链（Chain-of-Thought, CoT）以及过程奖励模型（PRM），成功模拟了“系统2”——即缓慢、深思熟虑、逻辑严密的推理过程【1】。
截至2024年中后期，行业主要遵循“缩放定律”，即通过增加模型参数量和训练数据量来换取性能提升。GPT-4o、Claude 3.5 Sonnet以及Llama 3等模型代表了这一路线的巅峰 【6】。为了在扩大参数规模的同时控制推理成本，2024年见证了混合专家模型（Mixture-of-Experts, MoE）的全面普及。【7】2024年的另一大主题是高质量人类文本数据的枯竭。随着互联网公开数据被挖掘殆尽，且面临版权和隐私的法律压力，开发者开始转向合成数据（Synthetic Data）。【8-9】
2025年的推理大模型技术进展主要集中在以下三个维度：
1.	推理模型的系统化（Systematization of Reasoning）： 不再依赖提示工程（Prompt Engineering），而是将推理能力内化为模型的核心训练目标。
2.	智能体的自进化（Self-Evolution of Agents）： 从依赖人工配置的静态智能体，转向能够通过环境反馈自主更新策略的动态系统。
3.	多模态与端侧的极致优化： 模型不仅要“聪明”，还要“全能”且“轻量”，能够理解物理世界并运行在边缘设备上。
下表总结了本年度最具代表性的模型及其核心技术特征，这些模型的共同点在于，它们都不再仅仅追求“预测下一个token”的准确率（Perplexity），而是追求生成过程的逻辑自洽性和最终结果的正确性。
表2.1：2025年的关键模型
模型名称	开发机构	核心特性	发布时间	关键技术突破
OpenAI o3 	OpenAI	推理增强，测试时计算（Test-time Compute）	2025年1月	强化学习思维链，AIME基准测试98.4%准确率【11】
DeepSeek-R1	DeepSeek-AI	开源推理，纯强化学习（Pure RL）	2025年1月	混合专家（MoE）架构，冷启动（Cold Start）数据策略【8】
Llama 4	Meta	通用基础，原生多模态	2025年4月	早期融合（Early Fusion）多模态架构，MoE设计【14,15】
Gemma 3	Google DeepMind	端侧多模态，超大词表	2025年	270M-27B参数，128k上下文，256k词表【16】
Confucius3-Math	网易有道	垂直领域数学推理	2025年6月	目标熵正则化，低成本RL后训练【9，18】
2025年，数学问题求解成为了检验LLM推理能力的核心战场。主要分为三个部分：
1.	过程奖励模型（Process Reward Models, PRM）。在复杂数学题中，模型可能通过错误的推理步骤偶然得到正确答案（False Positive），或者因为最后一步的计算失误而导致整个推理过程被否定（False Negative）。PRM能够对推理过程中的每一个步骤（Step）进行打分。此外，模型在推理时可以执行“最佳N次采样”（Best-of-N）或更复杂的树搜索。如果某一步骤的得分过低，模型会立即回溯并尝试其他路径，而不是一条道走到黑【26-27】。
2.	蒙特卡洛树搜索（MCTS）。2025年的创新在于将MCTS引入训练循环。通过在训练过程中使用MCTS来评估样本的难度，并筛选出高价值的训练子集进行强化微调（RFT），研究人员成功提升了模型在极难数学问题上的泛化能力 【21】
3. 形式化验证与神经符号系统。为了彻底解决数学推理中的“幻觉”问题，2025年的前沿研究开始尝试将LLM与形式化数学语言（如Lean 4）结合。模型不仅输出自然语言的解题过程，还尝试将其转化为Lean 4代码（Auto-formalization）。利用Lean 4编译器作为绝对客观的验证器。如果代码编译通过且证明成立，则该推理过程被标记为100%正确。这种方法首次实现了对LLM生成的数学推理轨迹的“零误判”验证，为训练数据的清洗提供了黄金标准【29】。
2.1.2.2 基于LLM的自进化Agent
如果说推理模型解决了“深度思考”的问题，那么自进化智能体（Self-Evolving Agents）则致力于解决“适应性”和“自主性”的问题【32，36】。不同于传统的基于固定提示词的Agent，自进化Agent拥有一个闭环的反馈更新机制。该架构包含四个核心组件：
	智能体系统（Agent System）： 执行任务的核心模型。
	环境（Environment）： 智能体交互的对象（如操作系统、IDE、金融市场）。
	系统输入（System Inputs）： 任务指令。
	优化器（Optimizer）：自进化的引擎，通常由另一个LLM充当。它观察智能体在环境中的表现，分析失败原因，并更新智能体的内部状态。
自进化的方式主要包括：
	基于奖励的进化（Reward-based Evolution）： 智能体根据任务完成的质量（如代码是否通过测试、股票交易是否盈利）获得奖励信号，并据此调整自身的提示词（Prompt Optimization）或工具使用策略【34】。
	模仿学习与经验回放（Imitation & Experience Replay）： 智能体会将自己成功的案例存入长时记忆，在遇到类似问题时检索并模仿过去的自己。
	群体进化（Population-based Evolution）： 多个智能体变体同时运行，通过“优胜劣汰”的机制，保留表现最好的智能体参数或配置【35】。

    '''

    desckuohao = '''字，硕士不少于2000字。）
基于大语言模型的复杂优化决策问题辅助建模与高效求解的相关文献综述，将从以下几个方面进行总结。
2.1 国内/外的研究现状及发展动态；
2.1.1 现有问题定义
在当前领域，多数工作[1,10,13,15,27,29]将运筹学问题的自动建模和求解定义为一个端到端任务。给定一个用自然语言表示的优化问题 𝑝，目标是生成一个数值答案 𝑜。本研究假设求解器的执行是绝对可靠的；也就是说，如果生成了正确的求解器代码，则结果被认为是正确的。
求解流程如下：给定一个问题 𝑝，LLM 首先构建一个数学模型，记为 𝑚 = LLM(𝑝)。随后，LLM 基于该模型生成可执行代码 𝑐 = LLM(𝑚)。该代码随后由 Python 解释器执行，产生数值输出 𝑜 = Python(𝑐)。为了进行评估，会将输出 𝑜 与真实答案 𝑔𝑡 进行比较，以计算准确率和执行通过率等指标。通过人工验证，当且仅当所有决策变量和目标值都正确匹配时，答案才被认为是正确的。简而言之，如果 𝑜 = 𝑔𝑡，则认为问题已解决。
    '''

    extract_numbers(desc)

# if __name__ == "__main__":
#     # 定义输入和输出文件名
#     # input_file = "sample_LP_100_Chinese.json"
#     # output_file = "sample_LP_100_Chinese.md"
#     for a in ["_A", "_B", "_C"]:
#         input_file_ = f"sample_LP_100_Chinese{a}.json"
#         output_file_ = f"sample_LP_100_Chinese{a}.md"
#         convert_json_to_md_with_math(input_file_, output_file_)