import gurobipy as gp
from gurobipy import GRB
import math

# 创建模型
m = gp.Model("Nonlinear_Production_Planning")

# 决策变量：产品产量（可以根据需要改为整数/非负整数，这里先设为连续非负）
x1 = m.addVar(lb=0.0, name="x1")  # Product I
x2 = m.addVar(lb=0.0, name="x2")  # Product II
x3 = m.addVar(lb=0.0, name="x3")  # Product III

# 辅助变量：各机器使用时间（可以直接用线性表达式，也可以显式建变量，这里建变量便于查看）
tA1 = m.addVar(lb=0.0, name="tA1")
tA2 = m.addVar(lb=0.0, name="tA2")
tB1 = m.addVar(lb=0.0, name="tB1")
tB2 = m.addVar(lb=0.0, name="tB2")

# 理论利润和非线性相关的中间变量
P    = m.addVar(lb=-GRB.INFINITY, name="P")      # 理论利润
P098 = m.addVar(lb=-GRB.INFINITY, name="P098")   # P^0.98
expP098 = m.addVar(lb=0.0, name="expP098")       # exp(P^0.98)
cos2P = m.addVar(lb=-1.0, ub=1.0, name="cos2P")  # cos(2P)
Z = m.addVar(lb=0.0, name="Z")                   # exp(P^0.98) + cos(2P)
F = m.addVar(lb=-GRB.INFINITY, name="F")         # 最终目标 ln(Z)

# -------------------------------------------------
# 1. 机器时间与产量的线性关系
# -------------------------------------------------
m.addConstr(tA1 == 5 * x1 + 10 * x2, name="Time_A1")
m.addConstr(tA2 == 7 * x1 + 9 * x2 + 12 * x3, name="Time_A2")
m.addConstr(tB1 == 6 * x1 + 8 * x2, name="Time_B1")
m.addConstr(tB2 == 4 * x1 + 11 * x3, name="Time_B2")

# 2. 机器可用时间约束
m.addConstr(tA1 <= 10000, name="Cap_A1")
m.addConstr(tA2 <= 4000,  name="Cap_A2")
m.addConstr(tB1 <= 7000,  name="Cap_B1")
m.addConstr(tB2 <= 4000,  name="Cap_B2")

# -------------------------------------------------
# 3. 理论利润 P 的表达式
# -------------------------------------------------

# 收入
R = 1.25 * x1 + 2.0 * x2 + 2.8 * x3

# 原材料成本
C_rm = 0.25 * x1 + 0.35 * x2 + 0.50 * x3

# 机器成本（按满负荷成本线性分摊）
cost_A1_per_hour = 321.0 / 10000.0
cost_A2_per_hour = 250.0 / 4000.0
cost_B1_per_hour = 783.0 / 7000.0
cost_B2_per_hour = 200.0 / 4000.0

C_mach = (cost_A1_per_hour * tA1 +
          cost_A2_per_hour * tA2 +
          cost_B1_per_hour * tB1 +
          cost_B2_per_hour * tB2)

# 理论利润 P = 收入 - 原材料成本 - 机器成本
m.addConstr(P == R - C_rm - C_mach, name="Profit_Def")

# -------------------------------------------------
# 4. 非线性部分:
#    P098 = P^0.98
#    expP098 = exp(P098)
#    cos2P = cos(2*P)
#    Z = expP098 + cos2P
#    F = ln(Z)
# -------------------------------------------------

# 允许非凸非线性建模
m.Params.NonConvex = 2

# P098 = P^0.98
m.addGenConstrPow(P, P098, 0.98, name="Pow_P_0_98")

# expP098 = exp(P098)
m.addGenConstrExp(P098, expP098, name="Exp_P098")

# cos2P = cos(2*P)
# 用一般非线性约束表示：cos2P = cos(2 * P)
m.addQConstr(cos2P == gp.cos(2 * P), name="Cos_2P")  # 如果你的 Gurobi 版本不支持 gp.cos(),
                                                    # 需改为通用函数回调或使用外部求值/近似

# Z = expP098 + cos2P
m.addConstr(Z == expP098 + cos2P, name="Z_Def")

# 保证对数里面大于 0 (略留一点安全裕度)
m.addConstr(Z >= 1e-6, name="Z_positive")

# F = ln(Z)
m.addGenConstrLog(Z, F, name="Log_Z")

# -------------------------------------------------
# 5. 目标函数：最大化 F
# -------------------------------------------------
m.setObjective(F, GRB.MAXIMIZE)

# 开始求解
m.optimize()

# 输出结果
if m.status == GRB.OPTIMAL or m.status == GRB.SUBOPTIMAL:
    print("\nOptimal (or best found) solution:")
    print(f"x1 (Product I)   = {x1.X:.4f}")
    print(f"x2 (Product II)  = {x2.X:.4f}")
    print(f"x3 (Product III) = {x3.X:.4f}")

    print(f"tA1 = {tA1.X:.4f}, tA2 = {tA2.X:.4f}, "
          f"tB1 = {tB1.X:.4f}, tB2 = {tB2.X:.4f}")

    print(f"Theoretical profit P = {P.X:.6f}")
    print(f"Final objective F = ln(exp(P^0.98)+cos(2P)) = {F.X:.6f}")
else:
    print("No optimal solution found. Status code:", m.status)
