import gurobipy as gp
from gurobipy import GRB

# 1. Define the model
model = gp.Model("TomatoOptimization")

# 2. Define parameters
# Total supply from the inspection report
total_tomatoes = 300000
grade_A_pct = 0.2
grade_B_pct = 0.7
grade_C_pct = 0.1

supply_A = total_tomatoes * grade_A_pct  # 60000 kg
supply_B = total_tomatoes * grade_B_pct  # 210000 kg
supply_C = total_tomatoes * grade_C_pct  # 30000 kg

# Product data from Table C-10 and Table C-11
kg_per_can_W = 1.8
kg_per_can_J = 2.0
kg_per_can_S = 2.5

demand_W = 800000
demand_J = 50000
demand_S = 80000

profit_per_can_W = 0.40
profit_per_can_J = 0.12
profit_per_can_S = 0.35
rd_sale_price = 0.001  # yuan/kg

# Quality scores
score_A = 9
score_B = 5
min_score_W = 8
min_score_J = 6
min_score_S = 4

# Expected scores for Grade C based on R&D sales threshold (y_R >= 20000)
# Case 1: y_R >= 20000 -> 80% success (score 4), 20% failure (score 1)
score_C_high = 0.8 * 4 + 0.2 * 1  # 3.4
# Case 2: y_R < 20000 -> 20% success (score 4), 80% failure (score 1)
score_C_low = 0.2 * 4 + 0.8 * 1   # 1.6

# 3. Create Decision Variables
x_W = model.addVar(lb=0, ub=demand_W, name="x_W")
x_J = model.addVar(lb=0, ub=demand_J, name="x_J")
x_S = model.addVar(lb=0, ub=demand_S, name="x_S")

# Material usage vars (Grade, Product)
y_AW = model.addVar(lb=0, name="y_AW")
y_BW = model.addVar(lb=0, name="y_BW")
y_AJ = model.addVar(lb=0, name="y_AJ")
y_BJ = model.addVar(lb=0, name="y_BJ")
y_AS = model.addVar(lb=0, name="y_AS")
y_BS = model.addVar(lb=0, name="y_BS")
y_CS = model.addVar(lb=0, name="y_CS")

# R&D Sale variable and its indicator
y_R = model.addVar(lb=0, name="y_R")
z_R = model.addVar(vtype=GRB.BINARY, name="z_R")

# Symbols provided in user context for consistency (set to 0)
y_CW = model.addVar(lb=0, ub=0, name="y_CW")
y_CJ = model.addVar(lb=0, ub=0, name="y_CJ")

# 4. Set up the Objective Function
# Maximize profit = profit from cans + revenue from Grade C sold to R&D
model.setObjective(profit_per_can_W * x_W + profit_per_can_J * x_J + profit_per_can_S * x_S + rd_sale_price * y_R, GRB.MAXIMIZE)

# 5. Add Constraints
# Supply Constraints
model.addConstr(y_AW + y_AJ + y_AS <= supply_A, name="Supply_A")
model.addConstr(y_BW + y_BJ + y_BS <= supply_B, name="Supply_B")
model.addConstr(y_CS + y_R <= supply_C, name="Supply_C")

# Raw Material Material Balance (Usage per Can)
model.addConstr(y_AW + y_BW == kg_per_can_W * x_W, name="Material_Whole")
model.addConstr(y_AJ + y_BJ == kg_per_can_J * x_J, name="Material_Juice")
model.addConstr(y_AS + y_BS + y_CS == kg_per_can_S * x_S, name="Material_Sauce")

# Quality constraints (Average score * total kg >= Required score * total kg)
model.addConstr(score_A * y_AW + score_B * y_BW >= min_score_W * (kg_per_can_W * x_W), name="Quality_Whole")
model.addConstr(score_A * y_AJ + score_B * y_BJ >= min_score_J * (kg_per_can_J * x_J), name="Quality_Juice")

# R&D Indicator Constraints
# If z_R = 1, then y_R >= 20000
model.addGenConstrIndicator(z_R, 1, y_R >= 20000, name="RD_Threshold_1")
# If z_R = 0, then y_R <= 19999.99
model.addGenConstrIndicator(z_R, 0, y_R <= 19999.99, name="RD_Threshold_0")

# Conditional Sauce Quality Constraints based on R&D Success probability
# If z_R = 1, expected score for Grade C is 3.4
model.addGenConstrIndicator(z_R, 1, score_A * y_AS + score_B * y_BS + score_C_high * y_CS >= min_score_S * (kg_per_can_S * x_S), name="Sauce_Qual_High")
# If z_R = 0, expected score for Grade C is 1.6
model.addGenConstrIndicator(z_R, 0, score_A * y_AS + score_B * y_BS + score_C_low * y_CS >= min_score_S * (kg_per_can_S * x_S), name="Sauce_Qual_Low")

# 6. Solve the model
model.optimize()

# 7. Print results
if model.status == GRB.OPTIMAL:
    print(f"Optimal Profit: {model.objVal:.2f}")
    print(f"FinalAnswer=【{model.objVal:.2f}】")
else:
    print("Optimal solution not found.")