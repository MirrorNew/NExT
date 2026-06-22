import matplotlib.pyplot as plt

# 沿用之前的消融实验阶段名称
conditions = [
    'Standard\n(Baseline)',
    '+ OR Knowledge\nPrompt',
    '+ Variable\nDecomposition'
]

# 原始错误率数据
error_rates = [57.58, 24.24, 13.63]

# 转换为准确率 (100% - Error Rate)
accuracy_rates = [round(100 - r, 2) for r in error_rates]

fig, ax = plt.subplots(figsize=(8, 6))

# 绘制折线图：
# marker='*' 表示星星符号
# markersize=18 适当放大星星尺寸使其显眼
# markerfacecolor='red' 和 markeredgecolor='red' 将星星设为红色
# color='#1f77b4' 保持折线主体为学术蓝色
ax.plot(conditions, accuracy_rates, marker='*', markersize=18, linestyle='-', linewidth=3,
        color='#1f77b4', markerfacecolor='red', markeredgecolor='red')

# 在数据点上方添加具体的百分比数值
for i, rate in enumerate(accuracy_rates):
    ax.annotate(f'{rate}%',
                (conditions[i], accuracy_rates[i]),
                textcoords="offset points",
                xytext=(0, 15), # 向上偏移以防遮挡星星
                ha='center',
                fontsize=12,
                fontweight='bold')

# 图表装饰
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Accuracy Improvement Trend on Non-linear Problems\n(Ablation Study on Prompting Strategies)',
             fontsize=14, pad=15, fontweight='bold')

# 因为是准确率，将Y轴的下限适当卡在20左右，上限留到100，让整体上升趋势更饱满
ax.set_ylim(40, 100)
ax.grid(True, linestyle='--', alpha=0.6)

# 调整布局并保存
fig.tight_layout()
plt.show()
# plt.savefig('accuracy_trend_line.png', dpi=300)