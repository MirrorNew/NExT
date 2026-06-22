import gurobipy as gp

# 1. Define parameters (from the provided list)
supply_A            = 6
supply_B            = 8
procurement_cost_A  = 9900
procurement_cost_B  = 6600
consumption         = {'A': {'I': 1, 'II': 2},
                       'B': {'I': 2, 'II': 1}}
unit_price_material = {'A': 9.9, 'B': 6.6}
wholesale_price     = {'I': 30, 'II': 20}
demand_upper_II     = 2000    # boxes
demand_diff_limit   = 1000    # boxes
demand_growth       = 0.25    # 25%

# Convert demand caps from boxes to thousand‐box units
demand_upper_II_tb   = demand_upper_II / 1000.0
demand_diff_limit_tb = demand_diff_limit / 1000.0

# -----------------------------
# Month 1: maximize wholesale revenue
# -----------------------------
model = gp.Model("Month1")

# Decision variables (thousand boxes)
x1 = model.addVar(lb=0, name="x1")  # Product I
x2 = model.addVar(lb=0, name="x2")  # Product II

# Objective: 30*x1 + 20*x2
model.setObjective(wholesale_price['I'] * x1 + wholesale_price['II'] * x2,
                   gp.GRB.MAXIMIZE)

# Constraints
model.addConstr(consumption['A']['I'] * x1 +
                consumption['A']['II'] * x2 <= supply_A,
                name="RawA_limit_M1")
model.addConstr(consumption['B']['I'] * x1 +
                consumption['B']['II'] * x2 <= supply_B,
                name="RawB_limit_M1")
model.addConstr(x2 <= demand_upper_II_tb,
                name="DemandCapII_M1")
model.addConstr(x2 - x1 <= demand_diff_limit_tb,
                name="IIvsI_ratio_M1")

# Solve Month 1
model.optimize()
z1     = model.ObjVal
x1_opt = x1.X
x2_opt = x2.X

# -----------------------------
# Month 2: maximize surplus = revenue - material cost
# -----------------------------
model = gp.Model("Month2")

# Decision variables (thousand boxes)
y1 = model.addVar(lb=0, name="y1")  # Product I
y2 = model.addVar(lb=0, name="y2")  # Product II

# Revenue and cost expressions
rev2  = wholesale_price['I'] * y1 + wholesale_price['II'] * y2
cost2 = (unit_price_material['A'] * (consumption['A']['I'] * y1 +
                                     consumption['A']['II'] * y2)
       + unit_price_material['B'] * (consumption['B']['I'] * y1 +
                                     consumption['B']['II'] * y2))

# Objective: maximize rev2 - cost2
model.setObjective(rev2 - cost2, gp.GRB.MAXIMIZE)

# Month 2 constraints
model.addConstr(y2 <= demand_upper_II_tb * (1 + demand_growth),
                name="DemandCapII_M2")
model.addConstr(y2 - y1 <= demand_diff_limit_tb,
                name="IIvsI_ratio_M2")
# Solve Month 2
model.optimize()
z2     = model.ObjVal
y1_opt = y1.X
y2_opt = y2.X

# Total profit for the two months
total_profit = z1 + z2

# Final answer output
print(f"FinalAnswer=【{total_profit:.4f}】")