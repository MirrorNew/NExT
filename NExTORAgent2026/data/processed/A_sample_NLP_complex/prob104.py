import gurobipy as gp
from gurobipy import GRB
import math

# 定义资产及其基准价值
# 注意：狗不能分开，按一个条目处理；钻石按三个条目处理
assets = {
    "Caillebotte Painting": 25000,
    "Diocletian Bust": 5000,
    "Yuan Vase": 20000,
    "Porsche 911": 40000,
    "Diamond 1": 12000,
    "Diamond 2": 12000,
    "Diamond 3": 12000,
    "Louis XV Sofa": 3000,
    "Jack Russell Dogs (Pair)": 6000,
    "AD 200 Sculpture": 10000,
    "Sailboat": 15000,
    "Harley-Davidson": 10000,
    "Cavour Furniture": 13000
}

asset_names = list(assets.keys())
prices = list(assets.values())
n_assets = len(prices)

# 创建模型
model = gp.Model("Estate_Allocation")

# --- 变量定义 ---
# x[i] = 1 表示分配给儿子 A，0 表示分配给儿子 B
x = model.addVars(n_assets, vtype=GRB.BINARY, name="x")

# 月份 t (1-12 整数)
t = model.addVar(lb=1, ub=12, vtype=GRB.INTEGER, name="t")

# 辅助变量用于处理 COS 函数
# theta = pi * t / 12
theta = model.addVar(lb=math.pi / 12, ub=math.pi, name="theta")
# cos_val = cos(theta)
cos_val = model.addVar(lb=-1, ub=1, name="cos_val")

# 目标函数：差异的绝对值
diff = model.addVar(lb=-GRB.INFINITY, name="diff")
abs_diff = model.addVar(lb=0, name="abs_diff")

# --- 约束条件 ---

# 1. 角度定义
model.addConstr(theta == (3.14159 * t) / 12, "theta_def")

# 2. 调用通用约束 API: cos_val = cos(theta)
model.addGenConstrCos(theta, cos_val, name="cos_constr")

# 3. 计算价值差异
# 波动系数 coeff = (1 + 0.1 * cos_val)
# 儿子 A 总值: sum(P_i * x_i * coeff)
# 儿子 B 总值: sum(P_i * (1 - x_i) * coeff)
# diff = (儿子 A - 儿子 B) = coeff * sum(P_i * (2*x_i - 1))
sum_expr = gp.quicksum(prices[i] * (2 * x[i] - 1) for i in range(n_assets))

# 由于存在变量相乘 (sum_expr * cos_val)，这变为二次约束
model.addConstr(diff == sum_expr * (1 + 0.1 * cos_val), "diff_def")

# 4. 取绝对值
model.addGenConstrAbs(abs_diff, diff, "abs_constr")

# --- 求解 ---
model.setObjective(abs_diff, GRB.MINIMIZE)

# 允许非凸优化 (因为存在变量乘法和三角函数)
model.setParam('NonConvex', 2)
model.optimize()

# --- 输出结果 ---
if model.status == GRB.OPTIMAL:
    print(f"\n最佳分配月份: {int(t.X)}")
    print(f"最小价值差异: ${abs_diff.X:.2f}")

    son_a = []
    son_b = []
    for i in range(n_assets):
        if x[i].X > 0.5:
            son_a.append(asset_names[i])
        else:
            son_b.append(asset_names[i])

    print("\n儿子 A 的资产:", son_a)
    print("儿子 B 的资产:", son_b)
else:
    print("未找到最优解")