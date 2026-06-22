import matplotlib.pyplot as plt
import numpy as np

# 数据准备
models = ['GPT-5', 'OPTIMUS\n(GPT-5)', 'LLMOPT']
categories = ['Category L', 'Category A', 'Category B', 'Category C']

# 准确率数据 (左图)
acc_data = {
    'GPT-5': [0.76, 0.27, 0.48, 0.59],
    'OPTIMUS(GPT-5)': [0.79, 0.18, 0.36, 0.43],
    'LLMOPT': [0.83, 0.02, 0.26, 0.45]
}

# 错误类型数据 (右图)
error_data = {
    'Type 1': [49.49, 22.63, 28.3],
    'Type 2': [32.54, 63.16, 52.34],
    'Type 3': [17.97, 14.21, 19.36]
}

# 设置画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# --- 左图：准确率折线图 ---
x = np.arange(len(categories))
markers = ['o', 's', '^']
colors_line = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, (model_name, scores) in enumerate(acc_data.items()):
    ax1.plot(categories, scores, marker=markers[i], linewidth=2.5,
             label=model_name, color=colors_line[i])
    for j, score in enumerate(scores):
        ax1.text(j, score + 0.02, f'{score}', ha='center', fontsize=10)

ax1.set_title('Accuracy by Category (Performance Cliff)', fontsize=14)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_ylim(0, 1.05)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

# --- 右图：错误类型水平堆叠图 ---
# 配色方案：低饱和度科学配色
# 顺序：蓝 (Type 1), 红 (Type 2), 绿 (Type 3)
color_list = ['#4E79A7', '#E15759', '#59A14F']
error_types = ['Type 1', 'Type 2', 'Type 3']

bar_width = 0.6
y_pos = np.arange(len(models))
lefts = np.zeros(len(models))  # 水平堆叠使用 left 累加

for i, err_type in enumerate(error_types):
    values = error_data[err_type]
    # 使用 barh 绘制水平条形图
    bars = ax2.barh(y_pos, values, bar_width, left=lefts,
                    label=err_type, color=color_list[i], alpha=0.9, edgecolor='white')

    # 添加数值标签
    for bar, val in zip(bars, values):
        width = bar.get_width()
        if width > 5:  # 只有宽度足够才显示文字
            ax2.text(bar.get_x() + width / 2,
                     bar.get_y() + bar.get_height() / 2,
                     f'{val}%',
                     ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    lefts += values

ax2.set_title('Error Type Distribution (Proportion)', fontsize=14)
ax2.set_xlabel('Percentage (%)', fontsize=12)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(models, fontsize=11)
ax2.set_xlim(0, 100)
# 图例放在下方
ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
ax2.grid(axis='x', linestyle='--', alpha=0.3)
ax2.invert_yaxis()  # 反转Y轴，让第一个模型显示在最上面

plt.tight_layout()
plt.show()