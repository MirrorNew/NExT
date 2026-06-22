# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np
#
# # 假设读取上面生成的 CSV
# # df = pd.read_csv("experiment_results_nonlinear.csv")
#
# # --- 构造模拟数据用于演示绘图效果 ---
# data = {
#     'Method': ['Standard']*4 + ['OptiMUS']*4 + ['LLMOPT']*4,
#     'NLtype': ['Linear', 'A', 'B', 'C'] * 3,
#     'Accuracy': [
#         0.76, 0.27, 0.48, 0.56,  # Standard: 线性很高，非线性全崩
#         0.79, 0.18, 0.36, 0.43,  # OptiMUS: 稍微好一点点
#         0.83, 0.02, 0.26, 0.45   # LLMOPT: 依然无法解决非线性
#     ]
# }
# df_viz = pd.DataFrame(data)
#
# # 设置风格
# sns.set_theme(style="whitegrid", font_scale=1.2)
# plt.rcParams['font.family'] = 'Times New Roman' # 论文常用字体
#
# # --- 画图 (a): 准确率折线图/柱状图 ---
# plt.figure(figsize=(10, 6))
#
# # 推荐使用带点折线图 (Line Plot with Markers) 来体现“趋势”和“断崖”
# sns.lineplot(data=df_viz, x='NLtype', y='Accuracy', hue='Method',
#              style='Method', markers=True, dashes=False, linewidth=2.5, markersize=10)
#
# # 或者使用柱状图 (Bar Plot) - 这种对比更强烈
# # sns.barplot(data=df_viz, x='NLtype', y='Accuracy', hue='Method', palette="viridis")
#
# plt.title('(a) Accuracy Comparison on Linear vs. Non-linear Problems', fontsize=16, pad=20)
# plt.ylabel('Accuracy (AC)', fontsize=14)
# plt.xlabel('Problem Type', fontsize=14)
# plt.ylim(0, 1.05)
# plt.legend(title='Method', loc='upper right')
#
# # 添加注释箭头，强调断崖
# plt.annotate('Performance Cliff', xy=(1, 0.4), xytext=(1.5, 0.7),
#              arrowprops=dict(facecolor='red', shrink=0.05),
#              fontsize=12, color='red')
#
# plt.tight_layout()
# plt.show()

# 第二次绘图
#############################
# 第二次绘图
#############################
# 第二次绘图
#############################
# 第二次绘图
#############################
# 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
#############################
# 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
############################## 第二次绘图
#############################




# import json
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# def calculate_error_rates(results_file, sample_file):
#     # 加载测试结果数据与样本数据
#     with open(results_file, "r", encoding='utf-8') as f:
#         results_data = json.load(f)
#
#     with open(sample_file, "r", encoding='utf-8') as f:
#         sample_data = json.load(f)
#
#     failed_cases = set(results_data.get("Failed cases", []))
#     stats = {}
#
#     for prob_key, prob_info in sample_data.items():
#         dataset_name = prob_info.get("index", "").split("_")[0]
#         is_linear = prob_info.get("NLtype") == "Linear"
#         type_str = "Linear" if is_linear else "Non-linear"
#
#         if dataset_name not in stats:
#             stats[dataset_name] = {
#                 "Linear": {"total": 0, "error": 0},
#                 "Non-linear": {"total": 0, "error": 0}
#             }
#
#         stats[dataset_name][type_str]["total"] += 1
#         if prob_key in failed_cases:
#             stats[dataset_name][type_str]["error"] += 1
#
#     error_rates = {}
#     for dataset, types in stats.items():
#         error_rates[dataset] = {}
#         for t in ["Linear", "Non-linear"]:
#             total = types[t]["total"]
#             error = types[t]["error"]
#             rate = (error / total) if total > 0 else 0
#             error_rates[dataset][t] = rate
#
#     return error_rates
#
#
# error_rates = calculate_error_rates(
#     "runs/【20260105_105135】_【ORSample_LABC_300】_【o4-mini】/AAA_result_300_264_182.txt",
#     "data/20251231_processedDATA/ORSample_LABC_300.json")
#
# # 绘图部分
# datasets = list(error_rates.keys())
# # 将比例乘以100转换为百分比形式
# linear_rates = [error_rates[d]["Linear"] * 100 for d in datasets]
# nonlinear_rates = [error_rates[d]["Non-linear"] * 100 for d in datasets]
#
# x = np.arange(len(datasets))
# width = 0.35
#
# fig, ax = plt.subplots(figsize=(10, 6))
# rects1 = ax.bar(x - width / 2, linear_rates, width, label='Linear', color='#5F86B0')
# rects2 = ax.bar(x + width / 2, nonlinear_rates, width, label='Non-linear', color='#CF3939')
#
# ax.set_ylabel('Error Rate (%)')
# ax.set_title('Error Rates by Dataset and Linearity')
# ax.set_xticks(x)
# ax.set_xticklabels(datasets, rotation=45, ha="right")
# ax.legend()
#
# # [新增] 在柱状图上方添加百分比标签，保留一位小数
# ax.bar_label(rects1, fmt='%.1f%%', padding=3)
# ax.bar_label(rects2, fmt='%.1f%%', padding=3)
#
# # [新增] 动态调整y轴上限，给顶部的百分比数字留出空间
# max_rate = max(max(linear_rates), max(nonlinear_rates))
# ax.set_ylim(0, max_rate + 15)
#
# fig.tight_layout()
# plt.savefig('error_rates_with_percentages.png')
# plt.show()
# plt.savefig('error_rates.png')



############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图############################# 第三次绘图
############################# 第三次绘图


# import matplotlib.pyplot as plt
# import numpy as np
#
# # 数据类别
# datasets = ['IndustryOR', 'LogiOR', 'NLP4LP']
# models = ['GPT-5', 'OPTIMUS', 'LLMOPT']
#
# # ！！！请在这里填入您统计出的三个模型的真实百分比数值！！！
# linear_data = {
#     'GPT-5': [23.33, 30.00, 8.00],
#     'OPTIMUS': [26.66, 35.00, 6.00],
#     'LLMOPT': [16.66, 25.00, 6.00]
# }
#
# nonlinear_data = {
#     'GPT-5': [57.58, 50.00, 45.24],
#     'OPTIMUS': [63.63, 64.00, 48.81],
#     'LLMOPT': [90.91, 80.00, 52.38]
# }
#
# x = np.arange(len(datasets))  # x轴刻度位置
# width = 0.25  # 每个柱子的宽度
#
# # 创建上下两个子图，nrows=2, ncols=1，并使用 sharex=True 共享X轴
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
#
# # 定义模型对应的颜色
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
#
# # ---------- 绘制上方图：非线性问题 (Non-linear) ----------
# multiplier = 0
# for i, (model, rates) in enumerate(linear_data.items()):
#     offset = width * multiplier
#     rects = ax1.bar(x + offset, rates, width, label=model, color=colors[i])
#     ax1.bar_label(rects, fmt='%.2f%%', padding=3, fontsize=10)
#     multiplier += 1
#
# ax1.set_ylabel('Error Rate (%)', fontsize=12)
# ax1.set_title('Linear Problem Error Rates', fontsize=14, pad=10)
# ax1.legend(loc='upper right', fontsize=11) # 图例放在上方图中即可
# ax1.grid(axis='y', linestyle='--', alpha=0.6)
#
# # ---------- 绘制下方图：线性问题 (Linear) ----------
# multiplier = 0
# for i, (model, rates) in enumerate(nonlinear_data.items()):
#     offset = width * multiplier
#     rects = ax2.bar(x + offset, rates, width, label=model, color=colors[i])
#     ax2.bar_label(rects, fmt='%.2f%%', padding=3, fontsize=10)
#     multiplier += 1
#
# ax2.set_ylabel('Error Rate (%)', fontsize=12)
# ax2.set_title('Non-linear Problem Error Rates', fontsize=14, pad=10)
# # 只在底部的图设置X轴标签，保持画面整洁
# ax2.set_xticks(x + width) # 刻度对齐到中间柱子
# ax2.set_xticklabels(datasets, fontsize=12)
# ax2.grid(axis='y', linestyle='--', alpha=0.6)
#
# # 动态调整各自的 y 轴上限，防止文字被遮挡
# max_rate_nonlinear = max([max(rates) for rates in nonlinear_data.values()])
# ax1.set_ylim(0, max_rate_nonlinear + 15)
#
# # max_rate_linear = max([max(rates) for rates in linear_data.values()])
# ax2.set_ylim(0, max_rate_nonlinear + 15)
#
# plt.subplots_adjust(hspace=0.15) # 调整上下图的间距
# fig.tight_layout()
#
# # 保存高质量图片
# plt.show()
# plt.savefig('stacked_multi_model_comparison.png', dpi=300)





import matplotlib.pyplot as plt
import numpy as np

# 1. 直接使用提供的错误率数据（无需再读取文件）
linear_data = {
    'GPT-5': [23.33, 30.00, 8.00],
    'OPTIMUS': [26.66, 35.00, 6.00],
    'LLMOPT': [16.66, 26.00, 6.00]
}

nonlinear_data = {
    'GPT-5': [57.58, 50.00, 45.24],
    'OPTIMUS': [63.63, 64.00, 48.81],
    'LLMOPT': [90.91, 80.00, 52.38]
}

# 2. 提取 GPT-5 的错误率数据
gpt_linear_error = [ (linear_data['GPT-5'][i]+linear_data['OPTIMUS'][i]+linear_data['LLMOPT'][i])/3.0
                     for i in range(3)]
gpt_nonlinear_error = [ (nonlinear_data['GPT-5'][i]+nonlinear_data['OPTIMUS'][i]+nonlinear_data['LLMOPT'][i])/3.0
                        for i in range(3)]

# 3. ！！！核心转换：计算准确率 (100 - 错误率) ！！！
gpt_linear_accuracy = [100 - e for e in gpt_linear_error]
gpt_nonlinear_accuracy = [100 - e for e in gpt_nonlinear_error]

datasets = ['IndustryOR', 'LogiOR', 'NLP4LP']
x = np.arange(len(datasets))
width = 0.35

# 4. 开始绘图
fig, ax = plt.subplots(figsize=(10, 6))

# 使用蓝色和绿色分别表示线性和非线性的准确率（正向指标使用绿色在视觉上更协调）
rects1 = ax.bar(x - width/2, gpt_linear_accuracy, width, label='Linear', color='#1f77b4')  # 经典蓝
rects2 = ax.bar(x + width/2, gpt_nonlinear_accuracy, width, label='Non-linear', color='#2ca02c')  # 经典绿

# 设置标签和标题，更改为 Accuracy Rate
ax.set_ylabel('Accuracy Rate (%)', fontsize=12)
ax.set_title('GPT-5 Accuracy Rates by Dataset and Linearity', fontsize=14, pad=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=12)
ax.legend(fontsize=11)

# 在柱状图上方添加百分比标签，保留两位小数
ax.bar_label(rects1, fmt='%.2f%%', padding=3, fontsize=11)
ax.bar_label(rects2, fmt='%.2f%%', padding=3, fontsize=11)

# 动态调整y轴上限，防止柱子上的文字被顶端边框遮挡
max_acc = max(max(gpt_linear_accuracy), max(gpt_nonlinear_accuracy))
ax.set_ylim(0, max_acc + 15)

# 添加水平网格线，使横向对比更加清晰
ax.grid(axis='y', linestyle='--', alpha=0.6)

fig.tight_layout()
# 导出为适合论文排版的高清格式
plt.show()
plt.savefig('gpt5_accuracy_rates.png', dpi=300)