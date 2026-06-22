import gurobipy as gp
from gurobipy import GRB

# ==============================
# 1. Import parameters (STRICTLY from Parameters List)
# ==============================
supply_A = 6
supply_B = 8
procurement_cost_A = 9900           # not directly used in objective (month 2 uses unit_price_material)
procurement_cost_B = 6600           # kept for completeness
consumption = {'A': {'I': 1, 'II': 2},
               'B': {'I': 2, 'II': 1}}
unit_price_material = {'A': 9.9, 'B': 6.6}
wholesale_price = {'I': 30, 'II': 20}
demand_upper_II = 2000              # boxes
demand_diff_limit = 1000            # boxes
demand_growth = 0.25

# Convert box-based demand parameters to thousand boxes
demand_upper_II_k = demand_upper_II / 1000.0       # 2000 boxes -> 2 thousand boxes
demand_diff_limit_k = demand_diff_limit / 1000.0   # 1000 boxes -> 1 thousand boxes

# Month 2 demand growth for product II
demand_upper_II_month2_k = demand_upper_II_k * (1 + demand_growth)  # 2 * 1.25 = 2.5 thousand boxes

# ==============================
# 2. Create Gurobi model
# ==============================
model = gp.Model("Tiantian_TwoMonth_Optimization")

# ==============================
# 3. Decision variables
# ==============================
# Month 1: production and sales (thousand boxes)
x1 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="x1")  # Product I, month 1
x2 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="x2")  # Product II, month 1

# Month 2: production and sales (thousand boxes)
y1 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="y1")  # Product I, month 2
y2 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="y2")  # Product II, month 2

model.update()

# ==============================
# 4. Objective function
#    Month 1: maximize wholesale revenue
#    Month 2: maximize surplus (revenue - raw material cost)
#    Overall: maximize sum of both (total two-month profit)
# ==============================

# Month 1 revenue (thousand yuan)
z1 = wholesale_price['I'] * x1 + wholesale_price['II'] * x2

# Month 2 revenue (thousand yuan)
rev2 = wholesale_price['I'] * y1 + wholesale_price['II'] * y2

# Month 2 raw material usage (tons)
A_usage_m2 = consumption['A']['I'] * y1 + consumption['A']['II'] * y2
B_usage_m2 = consumption['B']['I'] * y1 + consumption['B']['II'] * y2

# Month 2 raw material cost (thousand yuan)
cost2 = unit_price_material['A'] * A_usage_m2 + unit_price_material['B'] * B_usage_m2

# Month 2 surplus
z2 = rev2 - cost2

# Total two-month profit
Z = z1 + z2

model.setObjective(Z, GRB.MAXIMIZE)

# ==============================
# 5. Constraints
# ==============================

# ---- Month 1: Raw material limits ----
# Raw A: 1*x1 + 2*x2 ≤ 6
model.addConstr(
    consumption['A']['I'] * x1 + consumption['A']['II'] * x2 <= supply_A,
    name="RawA_M1"
)

# Raw B: 2*x1 + 1*x2 ≤ 8
model.addConstr(
    consumption['B']['I'] * x1 + consumption['B']['II'] * x2 <= supply_B,
    name="RawB_M1"
)

# ---- Month 1: Demand constraints ----
# Demand cap II (month 1): x2 ≤ 2 (thousand boxes)
model.addConstr(x2 <= demand_upper_II_k, name="DemandCapII_M1")

# II vs I ratio (month 1): x2 - x1 ≤ 1 (thousand boxes)
model.addConstr(x2 - x1 <= demand_diff_limit_k, name="II_vs_I_M1")

# ---- Month 2: Demand constraints ----
# Demand cap II (month 2): y2 ≤ 2.5 (thousand boxes)
model.addConstr(y2 <= demand_upper_II_month2_k, name="DemandCapII_M2")

# II vs I ratio (month 2): y2 - y1 ≤ 1 (thousand boxes)
model.addConstr(y2 - y1 <= demand_diff_limit_k, name="II_vs_I_M2")

# (No indicator constraints appear in this model; therefore none are added.)

# ==============================
# 6. Solve model
# ==============================
model.optimize()

# ==============================
# 7. Output results
# ==============================
if model.status == GRB.OPTIMAL:
    print(f"Optimal total 2-month profit (thousand yuan): {model.objVal:.4f}")
    print("Optimal production/sales plan (thousand boxes):")
    print(f"  x1 (Product I, Month 1) = {x1.X:.4f}")
    print(f"  x2 (Product II, Month 1) = {x2.X:.4f}")
    print(f"  y1 (Product I, Month 2) = {y1.X:.4f}")
    print(f"  y2 (Product II, Month 2) = {y2.X:.4f}")

    # The question asks for the total profit for the two months.
    FinalAnswer = model.objVal
    print(f"FinalAnswer=【{FinalAnswer}】")
else:
    print("No optimal solution found.")
    print("FinalAnswer=【None】")