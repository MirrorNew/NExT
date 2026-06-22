import gurobipy as gp
from gurobipy import GRB

# Create the model
model = gp.Model("HongMudanOptimization")

# --- Parameters ---
# Total Supply (from Parameters List)
total_tomatoes = 300000
supply_A = total_tomatoes * 0.2  # 60,000 kg
supply_B = total_tomatoes * 0.7  # 210,000 kg
supply_C = total_tomatoes * 0.1  # 30,000 kg

# Product parameters (Profit, Usage, Limits, Quality Scores)
# Whole Tomatoes
profit_W = 0.40
usage_W = 1.8
limit_W = 800000
min_score_W = 8

# Tomato Juice
profit_J = 0.12
usage_J = 2.0
limit_J = 50000
min_score_J = 6

# Tomato Sauce
profit_S = 0.35
usage_S = 2.5
limit_S = 80000
min_score_S = 4

# R&D Parameters
price_R = 0.001
threshold_R = 20000

# Scores
score_A = 9
score_B = 5

# Calculated expected scores for Grade C based on R&D sales volume
# If sales > 20000 (z_R=1): Score = 0.8*4 + 0.2*1 = 3.4
score_C_high = 0.8 * 4 + 0.2 * 1
# If sales <= 20000 (z_R=0): Score = 0.2*4 + 0.8*1 = 1.6
score_C_low = 0.2 * 4 + 0.8 * 1

# --- Decision Variables ---

# Production quantities (Cans)
x_W = model.addVar(lb=0, ub=limit_W, vtype=GRB.CONTINUOUS, name="x_W")
x_J = model.addVar(lb=0, ub=limit_J, vtype=GRB.CONTINUOUS, name="x_J")
x_S = model.addVar(lb=0, ub=limit_S, vtype=GRB.CONTINUOUS, name="x_S")

# Raw material allocation (kg)
# Grade A usage
y_AW = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y_AW")
y_AJ = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y_AJ")
y_AS = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y_AS")

# Grade B usage
y_BW = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y_BW")
y_BJ = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y_BJ")
y_BS = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y_BS")

# Grade C usage (Only for Sauce and R&D)
y_CS = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y_CS")
y_R = model.addVar(lb=0, ub=supply_C, vtype=GRB.CONTINUOUS, name="y_R")

# Binary variable for R&D sales threshold
# z_R = 1 if y_R >= 20000, else 0
z_R = model.addVar(vtype=GRB.BINARY, name="z_R")

# Auxiliary variable for Grade C quality contribution to Sauce
# This variable represents (Actual_Score_C * y_CS) and depends on z_R
Q_CS = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Q_CS")

# --- Objective Function ---
# Maximize Total Profit: Product Profit + R&D Sales Revenue
model.setObjective(
    profit_W * x_W + 
    profit_J * x_J + 
    profit_S * x_S + 
    price_R * y_R, 
    GRB.MAXIMIZE
)

# --- Constraints ---

# 1. Supply Constraints
model.addConstr(y_AW + y_AJ + y_AS <= supply_A, "Supply_A_Limit")
model.addConstr(y_BW + y_BJ + y_BS <= supply_B, "Supply_B_Limit")
model.addConstr(y_CS + y_R <= supply_C, "Supply_C_Limit")

# 2. Recipe/Production Balance Constraints
model.addConstr(y_AW + y_BW == usage_W * x_W, "Recipe_Whole")
model.addConstr(y_AJ + y_BJ == usage_J * x_J, "Recipe_Juice")
model.addConstr(y_AS + y_BS + y_CS == usage_S * x_S, "Recipe_Sauce")

# 3. Quality Constraints
# Whole Tomatoes: Avg Score >= 8
model.addConstr(score_A * y_AW + score_B * y_BW >= min_score_W * (y_AW + y_BW), "Quality_Whole")

# Tomato Juice: Avg Score >= 6
model.addConstr(score_A * y_AJ + score_B * y_BJ >= min_score_J * (y_AJ + y_BJ), "Quality_Juice")

# Tomato Sauce: Avg Score >= 4
# The score contribution of C (Q_CS) is handled via indicator constraints below
model.addConstr(score_A * y_AS + score_B * y_BS + Q_CS >= min_score_S * (y_AS + y_BS + y_CS), "Quality_Sauce")

# 4. Indicator Constraints for R&D Logic
# Link z_R to y_R amount
# If z_R = 1, then y_R >= 20000
model.addGenConstrIndicator(z_R, 1, y_R >= threshold_R, name="Link_zR_High")
# If z_R = 0, then y_R <= 20000
model.addGenConstrIndicator(z_R, 0, y_R <= threshold_R, name="Link_zR_Low")

# Link Q_CS (Quality contribution of C) to z_R and y_CS
# If z_R = 1 (High Success), C Score is 3.4
model.addGenConstrIndicator(z_R, 1, Q_CS == score_C_high * y_CS, name="Link_Quality_C_High")
# If z_R = 0 (Low Success), C Score is 1.6
model.addGenConstrIndicator(z_R, 0, Q_CS == score_C_low * y_CS, name="Link_Quality_C_Low")

# --- Solve ---
model.optimize()

# --- Output ---
if model.status == GRB.OPTIMAL:
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("FinalAnswer=【No optimal solution found】")