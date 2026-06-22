import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
# 假设我们只分析 LLMOPT 在 A, B, C 类上的错误分布
error_data = {
    'NLtype': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'ErrorType': [
        'Type 1: Modeling Semantic', 'Type 2: Non-linear Crash', 'Type 3: Coding Error',
        'Type 1: Modeling Semantic', 'Type 2: Non-linear Crash', 'Type 3: Coding Error',
        'Type 1: Modeling Semantic', 'Type 2: Non-linear Crash', 'Type 3: Coding Error'
    ],
    'Count': [
        10, 60, 5,   # A类: 大部分是 Type 2 (因为幂次导致Gurobi报错)
        15, 55, 5,   # B类: 大部分是 Type 2 (除法导致报错)
        30, 40, 5    # C类: 逻辑错误可能多一点，但Type 2依然很高
    ]
}
df_error = pd.DataFrame(error_data)

# 计算百分比
df_error['Percentage'] = df_error.groupby('NLtype')['Count'].transform(lambda x: x / x.sum() * 100)

# --- 画图 (b): 堆叠柱状图 ---
plt.figure(figsize=(10, 6))

# Pivot data for stacked bar chart
df_pivot = df_error.pivot(index='NLtype', columns='ErrorType', values='Percentage')

# 颜色映射：Type 2 用醒目的颜色（如橙色/红色）
colors = ['#a1c9f4', '#ff9f9b', '#d0bbff'] # 蓝(语义), 红(非线性), 紫(代码)
ax = df_pivot.plot(kind='bar', stacked=True, color=colors, figsize=(10, 6), width=0.6)

plt.title('(b) Error Distribution Analysis on Non-linear Problems', fontsize=16, pad=20)
plt.ylabel('Percentage of Errors (%)', fontsize=14)
plt.xlabel('Non-linear Type', fontsize=14)
plt.xticks(rotation=0)
plt.legend(title='Error Category', bbox_to_anchor=(1.05, 1), loc='upper left')

# 在柱子上标数值
for c in ax.containers:
    ax.bar_label(c, fmt='%.0f%%', label_type='center', color='black')

plt.tight_layout()
plt.show()