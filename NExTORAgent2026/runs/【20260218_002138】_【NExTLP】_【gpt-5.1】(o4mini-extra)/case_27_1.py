import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters and data
# =========================

# Use ONLY the provided Parameters List
total_tomatoes = 300000
grade_A_pct = 0.2
grade_B_pct = 0.7
grade_C_pct = 0.1
grade_C_initial_score = 1
sauce_source_grades = ['B', 'C']
grade_C_research_sale_price = 0.001
research_sales_threshold1 = 20000
research_success_prob1 = 0.8
research_post_score1 = 4
research_unsold_threshold = 10000
research_success_prob2 = 0.2
contract_A_price = 0.6
avg_score_A = 9
avg_score_B = 5
min_avg_score_whole = 8
min_avg_score_juice = 6
min_avg_score_sauce = 4
sauce_possible_source = ['B']
max_whole_cans_quality = 44500
extra_purchase_A = 80000
extra_purchase_price_A = 0.85

Table_C_10 = [
    {'Product': 'Whole tomatoes', 'Sales price': 4.0, 'Raw material usage': 1.8, 'Demand forecast': 800000},
    {'Product': 'Tomato juice',  'Sales price': 4.5, 'Raw material usage': 2.0, 'Demand forecast': 50000},
    {'Product': 'Tomato sauce',  'Sales price': 3.8, 'Raw material usage': 2.5, 'Demand forecast': 80000}
]

Table_C_11 = [
    {'Cost Category': 'Sales Price',               'Whole Tomatoes': 4.0,  'Tomato Juice': 4.5,  'Tomato Sauce': 3.8},
    {'Cost Category': 'Direct Labor',              'Whole Tomatoes': 1.18, 'Tomato Juice': 1.32, 'Tomato Sauce': 0.54},
    {'Cost Category': 'Variable Management Costs', 'Whole Tomatoes': 0.24, 'Tomato Juice': 0.36, 'Tomato Sauce': 0.26},
    {'Cost Category': 'Variable Cost of Sales',    'Whole Tomatoes': 0.4,  'Tomato Juice': 0.85, 'Tomato Sauce': 0.38},
    {'Cost Category': 'Packaging Materials',       'Whole Tomatoes': 0.7,  'Tomato Juice': 0.65, 'Tomato Sauce': 0.77},
    {'Cost Category': 'Raw Materials',             'Whole Tomatoes': 1.08, 'Tomato Juice': 1.2,  'Tomato Sauce': 1.5},
    {'Cost Category': 'Subtotal Variable Costs',   'Whole Tomatoes': 3.6,  'Tomato Juice': 4.38, 'Tomato Sauce': 3.45},
    {'Cost Category': 'Net Profit Per Can',        'Whole Tomatoes': 0.4,  'Tomato Juice': 0.12, 'Tomato Sauce': 0.35}
]

# Derived supplies from total_tomatoes and percentages
grade_A_supply = total_tomatoes * grade_A_pct       # 300000 * 0.2 = 60000
grade_B_supply = total_tomatoes * grade_B_pct       # 300000 * 0.7 = 210000
grade_C_supply = total_tomatoes * grade_C_pct       # 300000 * 0.1 = 30000

# Demand forecasts and raw material usages (from Table_C_10)
raw_usage_whole = Table_C_10[0]['Raw material usage']      # 1.8
raw_usage_juice = Table_C_10[1]['Raw material usage']      # 2.0
raw_usage_sauce = Table_C_10[2]['Raw material usage']      # 2.5

demand_whole = Table_C_10[0]['Demand forecast']            # 800000
demand_juice = Table_C_10[1]['Demand forecast']            # 50000
demand_sauce = Table_C_10[2]['Demand forecast']            # 80000

# Net profits per can (from Table_C_11 last row)
net_profit_whole = Table_C_11[-1]['Whole Tomatoes']        # 0.4
net_profit_juice = Table_C_11[-1]['Tomato Juice']          # 0.12
net_profit_sauce = Table_C_11[-1]['Tomato Sauce']          # 0.35

# Research sale price for C-grade tomatoes
price_research_C = grade_C_research_sale_price            # 0.001

# =========================
# 2. Create model
# =========================
model = gp.Model("Hong_Mudan_Tomato_Production")

# =========================
# 3. Decision variables
# =========================

# Product quantities (cans)
x_W = model.addVar(lb=0.0, ub=demand_whole, vtype=GRB.CONTINUOUS, name="x_W")  # whole tomatoes
x_J = model.addVar(lb=0.0, ub=demand_juice, vtype=GRB.CONTINUOUS, name="x_J")  # tomato juice
x_S = model.addVar(lb=0.0, ub=demand_sauce, vtype=GRB.CONTINUOUS, name="x_S")  # tomato sauce

# Raw material allocations (kg)
y_AW = model.addVar(lb=0.0, ub=grade_A_supply, vtype=GRB.CONTINUOUS, name="y_AW")
y_BW = model.addVar(lb=0.0, ub=grade_B_supply, vtype=GRB.CONTINUOUS, name="y_BW")
y_AJ = model.addVar(lb=0.0, ub=grade_A_supply, vtype=GRB.CONTINUOUS, name="y_AJ")
y_BJ = model.addVar(lb=0.0, ub=grade_B_supply, vtype=GRB.CONTINUOUS, name="y_BJ")
y_AS = model.addVar(lb=0.0, ub=grade_A_supply, vtype=GRB.CONTINUOUS, name="y_AS")
y_BS = model.addVar(lb=0.0, ub=grade_B_supply, vtype=GRB.CONTINUOUS, name="y_BS")
y_CS = model.addVar(lb=0.0, ub=grade_C_supply, vtype=GRB.CONTINUOUS, name="y_CS")

# C-grade tomatoes sold to research department (kg)
y_R = model.addVar(lb=0.0, ub=grade_C_supply, vtype=GRB.CONTINUOUS, name="y_R")

# Binary indicator for research volume regime
z_R = model.addVar(vtype=GRB.BINARY, name="z_R")

model.update()

# =========================
# 4. Objective function
# =========================

# Maximize total profit:
# Z = 0.40·x_W + 0.12·x_J + 0.35·x_S + 0.001·y_R
obj_expr = (
    net_profit_whole * x_W +
    net_profit_juice * x_J +
    net_profit_sauce * x_S +
    price_research_C * y_R
)

model.setObjective(obj_expr, GRB.MAXIMIZE)

# =========================
# 5. Constraints
# =========================

# --- Supply constraints for each grade ---
# Grade A supply: y_AW + y_AJ + y_AS ≤ 60000
model.addConstr(y_AW + y_AJ + y_AS <= grade_A_supply, name="Grade_A_supply")

# Grade B supply: y_BW + y_BJ + y_BS ≤ 210000
model.addConstr(y_BW + y_BJ + y_BS <= grade_B_supply, name="Grade_B_supply")

# Grade C balance: y_CS + y_R ≤ 30000
model.addConstr(y_CS + y_R <= grade_C_supply, name="Grade_C_balance")

# --- C-only-for-sauce ---
# y_CW = 0; y_CJ = 0 are modeled by not creating those variables at all

# --- Raw material balance for products ---
# Whole tomatoes: y_AW + y_BW = 1.8·x_W
model.addConstr(y_AW + y_BW == raw_usage_whole * x_W, name="Raw_whole_balance")

# Tomato juice: y_AJ + y_BJ = 2.0·x_J
model.addConstr(y_AJ + y_BJ == raw_usage_juice * x_J, name="Raw_juice_balance")

# Tomato sauce: y_AS + y_BS + y_CS = 2.5·x_S
model.addConstr(y_AS + y_BS + y_CS == raw_usage_sauce * x_S, name="Raw_sauce_balance")

# --- Demand bounds (already in variable upper bounds, but also added as constraints) ---
model.addConstr(x_W <= demand_whole, name="Demand_whole")
model.addConstr(x_J <= demand_juice, name="Demand_juice")
model.addConstr(x_S <= demand_sauce, name="Demand_sauce")

# --- Quality constraints ---
# Quality for whole: 9·y_AW + 5·y_BW ≥ 8·(y_AW + y_BW)
model.addConstr(
    avg_score_A * y_AW + avg_score_B * y_BW >=
    min_avg_score_whole * (y_AW + y_BW),
    name="Quality_whole"
)

# Quality for juice: 9·y_AJ + 5·y_BJ ≥ 6·(y_AJ + y_BJ)
model.addConstr(
    avg_score_A * y_AJ + avg_score_B * y_BJ >=
    min_avg_score_juice * (y_AJ + y_BJ),
    name="Quality_juice"
)

# Quality for sauce: 9·y_AS + 5·y_BS + 1·y_CS ≥ 4·(y_AS + y_BS + y_CS)
model.addConstr(
    avg_score_A * y_AS + avg_score_B * y_BS + grade_C_initial_score * y_CS >=
    min_avg_score_sauce * (y_AS + y_BS + y_CS),
    name="Quality_sauce"
)

# --- R&D threshold constraints using indicator constraints ---
# R&D threshold lower bound: y_R ≥ 20000·z_R
# R&D threshold upper bound: y_R ≤ 20000 + 10000·z_R
# Implement with addGenConstrIndicator for both regimes

# If z_R == 1 then y_R >= 20000
model.addGenConstrIndicator(z_R, 1, y_R >= research_sales_threshold1, name="Ind_R_lower")

# If z_R == 0 then y_R <= 20000
model.addGenConstrIndicator(z_R, 0, y_R <= research_sales_threshold1, name="Ind_R_upper")

# Also enforce the looser linear inequalities explicitly described:
model.addConstr(y_R >= research_sales_threshold1 * z_R, name="R_linear_lower")
model.addConstr(
    y_R <= research_sales_threshold1 + research_unsold_threshold * z_R,
    name="R_linear_upper"
)

# =========================
# 6. Optimize
# =========================
model.optimize()

# =========================
# 7. Print results
# =========================
if model.status == GRB.OPTIMAL:
    print(f"Optimal objective (max profit): {model.objVal:.6f}")
    print(f"x_W (whole, cans) = {x_W.X:.4f}")
    print(f"x_J (juice, cans) = {x_J.X:.4f}")
    print(f"x_S (sauce, cans) = {x_S.X:.4f}")
    print(f"y_AW (A->whole, kg) = {y_AW.X:.4f}")
    print(f"y_BW (B->whole, kg) = {y_BW.X:.4f}")
    print(f"y_AJ (A->juice, kg) = {y_AJ.X:.4f}")
    print(f"y_BJ (B->juice, kg) = {y_BJ.X:.4f}")
    print(f"y_AS (A->sauce, kg) = {y_AS.X:.4f}")
    print(f"y_BS (B->sauce, kg) = {y_BS.X:.4f}")
    print(f"y_CS (C->sauce, kg) = {y_CS.X:.4f}")
    print(f"y_R (C sold to R&D, kg) = {y_R.X:.4f}")
    print(f"z_R (research indicator) = {z_R.X:.0f}")

    # FinalAnswer is the maximum profit
    FinalAnswer = model.objVal
else:
    FinalAnswer = float('nan')

print(f"FinalAnswer=【{FinalAnswer}】")