import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Define parameters (from Parameters List)
# =========================
products = ['I', 'II']
supply_month1_A_t = 6
supply_month1_B_t = 8
cost_A_yuan_per_t = 9900
cost_B_yuan_per_t = 6600
cost_A_thousand_yuan_per_t = 9.9
cost_B_thousand_yuan_per_t = 6.6
consumption_A = {'I': 1, 'II': 2}
consumption_B = {'I': 2, 'II': 1}
max_sales_month1_II_boxes = 2000
max_diff_II_minus_I_boxes = 1000
demand_growth_rate_month2 = 0.25
wholesale_price_thousand_yuan_per_thousand_boxes = {'I': 30, 'II': 20}
wholesale_price_yuan_per_box = {'I': 30, 'II': 20}

# Convert box-based parameters to thousand-box units where needed
max_sales_month1_II_thousand_boxes = max_sales_month1_II_boxes / 1000.0
max_diff_II_minus_I_thousand_boxes = max_diff_II_minus_I_boxes / 1000.0

# Precompute per-thousand-box raw material cost in thousand yuan
# Product I: A=1t, B=2t
cost_per_k_thousand_yuan = {}
cost_per_k_thousand_yuan['I'] = (
    cost_A_thousand_yuan_per_t * consumption_A['I']
    + cost_B_thousand_yuan_per_t * consumption_B['I']
)
# Product II: A=2t, B=1t
cost_per_k_thousand_yuan['II'] = (
    cost_A_thousand_yuan_per_t * consumption_A['II']
    + cost_B_thousand_yuan_per_t * consumption_B['II']
)

# Net surplus coefficients per thousand boxes (thousand yuan)
net_profit_coef = {}
for p in products:
    net_profit_coef[p] = (
        wholesale_price_thousand_yuan_per_thousand_boxes[p]
        - cost_per_k_thousand_yuan[p]
    )

# =========================
# 2. Month 1 model: maximize revenue
# =========================
model = gp.Model("Tiantian_Month1_Revenue_Max")

# Decision variables (thousand boxes)
x1 = model.addVar(lb=0.0, name="x1")  # Product I month 1
y1 = model.addVar(lb=0.0, name="y1")  # Product II month 1

# Auxiliary variables section (none required for this linear model,
# but we declare placeholders to comply with structure)
aux_vars = {}
# Example of how auxiliary vars would be declared:
# aux_vars['something'] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="aux_something")

model.update()

# Objective: maximize revenue in month 1 (thousand yuan)
model.setObjective(
    wholesale_price_thousand_yuan_per_thousand_boxes['I'] * x1
    + wholesale_price_thousand_yuan_per_thousand_boxes['II'] * y1,
    GRB.MAXIMIZE
)

# Constraints: raw material supply (tons)
model.addConstr(
    consumption_A['I'] * x1 + consumption_A['II'] * y1 <= supply_month1_A_t,
    name="Month1_A_supply",
)
model.addConstr(
    consumption_B['I'] * x1 + consumption_B['II'] * y1 <= supply_month1_B_t,
    name="Month1_B_supply",
)

# Demand and relational constraints in month 1
model.addConstr(
    y1 <= max_sales_month1_II_thousand_boxes,
    name="Month1_ProdII_demand",
)
model.addConstr(
    y1 - x1 <= max_diff_II_minus_I_thousand_boxes,
    name="Month1_II_not_exceed_I_by_1k",
)

# Nonnegativity is already enforced by lb=0, but we name them explicitly
model.addConstr(x1 >= 0, name="Month1_nonnegativity_I")
model.addConstr(y1 >= 0, name="Month1_nonnegativity_II")

# Optimize month 1 model
model.optimize()

if model.Status != GRB.OPTIMAL:
    raise RuntimeError("Month 1 model did not solve to optimality.")

x1_star = x1.X
y1_star = y1.X

# =========================
# 3. Month 2 model: maximize surplus given month 1 optimal solution
# =========================
model2 = gp.Model("Tiantian_Month2_Surplus_Max")

# Decision variables (thousand boxes)
x2 = model2.addVar(lb=0.0, name="x2")  # Product I month 2
y2 = model2.addVar(lb=0.0, name="y2")  # Product II month 2

# Auxiliary variables section for month 2 (none actually needed)
aux_vars2 = {}

model2.update()

# Month 2 demand upper bounds (25% growth over month 1 optimal quantities)
max_x2 = (1.0 + demand_growth_rate_month2) * x1_star
max_y2 = (1.0 + demand_growth_rate_month2) * y1_star

model2.addConstr(x2 <= max_x2, name="Month2_I_demand")
model2.addConstr(y2 <= max_y2, name="Month2_II_demand")

model2.addConstr(x2 >= 0, name="Month2_nonnegativity_I")
model2.addConstr(y2 >= 0, name="Month2_nonnegativity_II")

# Objective: maximize surplus in month 2 (thousand yuan)
model2.setObjective(
    net_profit_coef['I'] * x2 + net_profit_coef['II'] * y2,
    GRB.MAXIMIZE
)

# Solve month 2 model
model2.optimize()

if model2.Status != GRB.OPTIMAL:
    raise RuntimeError("Month 2 model did not solve to optimality.")

x2_star = x2.X
y2_star = y2.X

# =========================
# 4. Compute total two-month profit (surplus) in thousand yuan
# =========================
# Month 1 surplus
month1_surplus = (
    net_profit_coef['I'] * x1_star + net_profit_coef['II'] * y1_star
)

# Month 2 surplus
month2_surplus = (
    net_profit_coef['I'] * x2_star + net_profit_coef['II'] * y2_star
)

S_total = month1_surplus + month2_surplus

# =========================
# 5. Print detailed results
# =========================
print("===== Month 1 Results =====")
print(f"x1 (Product I, thousand boxes): {x1_star:.6f}")
print(f"y1 (Product II, thousand boxes): {y1_star:.6f}")
print(f"Month 1 revenue (thousand yuan): "
      f"{wholesale_price_thousand_yuan_per_thousand_boxes['I'] * x1_star + wholesale_price_thousand_yuan_per_thousand_boxes['II'] * y1_star:.6f}")
print(f"Month 1 surplus (thousand yuan): {month1_surplus:.6f}")

print("\n===== Month 2 Results =====")
print(f"x2 (Product I, thousand boxes): {x2_star:.6f}")
print(f"y2 (Product II, thousand boxes): {y2_star:.6f}")
print(f"Month 2 surplus (thousand yuan): {month2_surplus:.6f}")

print("\n===== Total Over Two Months =====")
print(f"Total surplus S_total (thousand yuan): {S_total:.6f}")

# =========================
# 6. Final answer in required format
# =========================
print(f"FinalAnswer=【{S_total}】")