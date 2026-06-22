import gurobipy as gp
from gurobipy import GRB

# ==============================
# 1. Define Parameters (from Parameters List)
# ==============================
total_tomatoes_kg = 300000
grade_A_fraction = 0.2
grade_B_fraction = 0.7
grade_C_fraction = 0.1
grade_C_score_initial = 1
grade_C_RnD_sale_price_yuan_per_kg = 0.001
RnD_success_prob_high = 0.8
RnD_success_prob_low = 0.2
RnD_sales_threshold_high_kg = 20000
RnD_unsold_threshold_low_kg = 10000
grade_C_score_after_RnD_success = 4
tomato_purchase_price_yuan_per_kg = 0.6
extra_grade_A_available_kg = 80000
extra_grade_A_price_yuan_per_kg = 0.85
grade_A_score = 9
grade_B_score = 5
min_quality_whole_tomatoes = 8
min_quality_tomato_juice = 6
min_quality_tomato_sauce = 4
max_whole_tomatoes_cans_approx = 44500
products = ['Whole tomatoes', 'Tomato juice', 'Tomato sauce']
sales_price_yuan_per_can = {
    'Whole tomatoes': 4.0,
    'Tomato juice': 4.5,
    'Tomato sauce': 3.8
}
raw_material_usage_kg_per_can = {
    'Whole tomatoes': 1.8,
    'Tomato juice': 2.0,
    'Tomato sauce': 2.5
}
demand_forecast_can_max = {
    'Whole tomatoes': 800000,
    'Tomato juice': 50000,
    'Tomato sauce': 80000
}
direct_labor_yuan_per_can = {
    'Whole tomatoes': 1.18,
    'Tomato juice': 1.32,
    'Tomato sauce': 0.54
}
variable_management_costs_yuan_per_can = {
    'Whole tomatoes': 0.24,
    'Tomato juice': 0.36,
    'Tomato sauce': 0.26
}
variable_selling_cost_yuan_per_can = {
    'Whole tomatoes': 0.4,
    'Tomato juice': 0.85,
    'Tomato sauce': 0.38
}
packaging_materials_yuan_per_can = {
    'Whole tomatoes': 0.7,
    'Tomato juice': 0.65,
    'Tomato sauce': 0.77
}
raw_material_cost_yuan_per_can = {
    'Whole tomatoes': 1.08,
    'Tomato juice': 1.2,
    'Tomato sauce': 1.5
}
subtotal_variable_costs_yuan_per_can = {
    'Whole tomatoes': 3.6,
    'Tomato juice': 4.38,
    'Tomato sauce': 3.45
}
net_profit_per_can_yuan = {
    'Whole tomatoes': 0.4,
    'Tomato juice': 0.12,
    'Tomato sauce': 0.35
}
Table_1_Demand_Forecast_and_Raw_Material_Usage = {
    'Whole tomatoes': {
        'sales_price_yuan_per_can': 4.0,
        'raw_material_usage_kg_per_can': 1.8,
        'demand_forecast_can_max': 800000
    },
    'Tomato juice': {
        'sales_price_yuan_per_can': 4.5,
        'raw_material_usage_kg_per_can': 2.0,
        'demand_forecast_can_max': 50000
    },
    'Tomato sauce': {
        'sales_price_yuan_per_can': 3.8,
        'raw_material_usage_kg_per_can': 2.5,
        'demand_forecast_can_max': 80000
    }
}
Table_2_Product_Profit_Analysis = {
    'Whole tomatoes': {
        'Sales Price': 4.0,
        'Direct Labor': 1.18,
        'Variable Management Costs': 0.24,
        'Variable Cost of Sales': 0.4,
        'Packaging Materials': 0.7,
        'Raw Materials': 1.08,
        'Subtotal Variable Costs': 3.6,
        'Net Profit Per Can': 0.4
    },
    'Tomato juice': {
        'Sales Price': 4.5,
        'Direct Labor': 1.32,
        'Variable Management Costs': 0.36,
        'Variable Cost of Sales': 0.85,
        'Packaging Materials': 0.65,
        'Raw Materials': 1.2,
        'Subtotal Variable Costs': 4.38,
        'Net Profit Per Can': 0.12
    },
    'Tomato sauce': {
        'Sales Price': 3.8,
        'Direct Labor': 0.54,
        'Variable Management Costs': 0.26,
        'Variable Cost of Sales': 0.38,
        'Packaging Materials': 0.77,
        'Raw Materials': 1.5,
        'Subtotal Variable Costs': 3.45,
        'Net Profit Per Can': 0.35
    }
}

# Derived supplies by grade
grade_A_supply_kg = int(total_tomatoes_kg * grade_A_fraction)  # 60,000
grade_B_supply_kg = int(total_tomatoes_kg * grade_B_fraction)  # 210,000
grade_C_supply_kg = int(total_tomatoes_kg * grade_C_fraction)  # 30,000

# ==============================
# 2. Create Model
# ==============================
model = gp.Model("Hong_Mudan_Tomato_Production")

# ==============================
# 3. Decision Variables
# ==============================

# Production quantities (cans)
x_W = model.addVar(lb=0.0, ub=demand_forecast_can_max['Whole tomatoes'],
                   vtype=GRB.CONTINUOUS, name="x_W")  # whole tomatoes
x_J = model.addVar(lb=0.0, ub=demand_forecast_can_max['Tomato juice'],
                   vtype=GRB.CONTINUOUS, name="x_J")  # tomato juice
x_S = model.addVar(lb=0.0, ub=demand_forecast_can_max['Tomato sauce'],
                   vtype=GRB.CONTINUOUS, name="x_S")  # tomato sauce

# Allocation of A and B to products (kg)
y_A_W = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="y_A_W")  # A to whole
y_B_W = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="y_B_W")  # B to whole

y_A_J = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="y_A_J")  # A to juice
y_B_J = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="y_B_J")  # B to juice

y_B_S = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="y_B_S")  # B to sauce
y_C_S = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="y_C_S")  # C to sauce

# C-grade sold to R&D and unused (kg)
z_C = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="z_C")   # sold to R&D

# Unused tomatoes by grade (kg)
u_A = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="u_A")
u_B = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="u_B")
u_C = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="u_C")

# ==============================
# 4. Objective Function
# ==============================
# max Z = 0.40·x_W + 0.12·x_J + 0.35·x_S + 0.001·z_C
obj = (
    net_profit_per_can_yuan['Whole tomatoes'] * x_W +
    net_profit_per_can_yuan['Tomato juice'] * x_J +
    net_profit_per_can_yuan['Tomato sauce'] * x_S +
    grade_C_RnD_sale_price_yuan_per_kg * z_C
)

model.setObjective(obj, GRB.MAXIMIZE)

# ==============================
# 5. Constraints
# ==============================

# (1) Supply balances
model.addConstr(
    y_A_W + y_A_J + u_A == grade_A_supply_kg,
    name="Total_A_supply"
)

model.addConstr(
    y_B_W + y_B_J + y_B_S + u_B == grade_B_supply_kg,
    name="Total_B_supply"
)

model.addConstr(
    y_C_S + z_C + u_C == grade_C_supply_kg,
    name="Total_C_supply"
)

# (2) Raw material balances for each product
# Whole tomatoes: y_A^W + y_B^W = 1.8 * x_W
model.addConstr(
    y_A_W + y_B_W == raw_material_usage_kg_per_can['Whole tomatoes'] * x_W,
    name="Whole_raw_balance"
)

# Tomato juice: y_A^J + y_B^J = 2.0 * x_J
model.addConstr(
    y_A_J + y_B_J == raw_material_usage_kg_per_can['Tomato juice'] * x_J,
    name="Juice_raw_balance"
)

# Tomato sauce: y_B^S + y_C^S = 2.5 * x_S
model.addConstr(
    y_B_S + y_C_S == raw_material_usage_kg_per_can['Tomato sauce'] * x_S,
    name="Sauce_raw_balance"
)

# (3) Demand bounds (already in variable bounds for x_J and x_S, add explicit for whole)
model.addConstr(
    x_W <= max_whole_tomatoes_cans_approx,
    name="Whole_max_from_quality"
)

# (4) Quality constraints
# Whole: 9·y_A^W + 5·y_B^W ≥ 8·(y_A^W + y_B^W)
model.addConstr(
    grade_A_score * y_A_W + grade_B_score * y_B_W >=
    min_quality_whole_tomatoes * (y_A_W + y_B_W),
    name="Whole_quality"
)

# Juice: 9·y_A^J + 5·y_B^J ≥ 6·(y_A^J + y_B^J)
model.addConstr(
    grade_A_score * y_A_J + grade_B_score * y_B_J >=
    min_quality_tomato_juice * (y_A_J + y_B_J),
    name="Juice_quality"
)

# Sauce quality is implicitly satisfied by using B (5) and improved C (>=4),
# so no extra explicit constraint is added here.

# (5) Nonnegativity implicitly enforced by lb=0 in variable definitions

# (6) Ignore extra purchase of A: no constraint added beyond fixed supply above

# ==============================
# 6. Optimize
# ==============================
model.optimize()

# ==============================
# 7. Print Results
# ==============================
if model.status == GRB.OPTIMAL:
    print("Optimal solution found.")
    print(f"Objective value (Max Profit) = {model.objVal:.4f} yuan")
    print("\nProduction Plan (cans):")
    print(f"  Whole tomatoes (x_W):  {x_W.X:.2f}")
    print(f"  Tomato juice (x_J):    {x_J.X:.2f}")
    print(f"  Tomato sauce (x_S):    {x_S.X:.2f}")

    print("\nAllocation of Grade A tomatoes (kg):")
    print(f"  To whole (y_A_W):      {y_A_W.X:.2f}")
    print(f"  To juice (y_A_J):      {y_A_J.X:.2f}")
    print(f"  Unused A (u_A):        {u_A.X:.2f}")

    print("\nAllocation of Grade B tomatoes (kg):")
    print(f"  To whole (y_B_W):      {y_B_W.X:.2f}")
    print(f"  To juice (y_B_J):      {y_B_J.X:.2f}")
    print(f"  To sauce (y_B_S):      {y_B_S.X:.2f}")
    print(f"  Unused B (u_B):        {u_B.X:.2f}")

    print("\nAllocation of Grade C tomatoes (kg):")
    print(f"  To sauce (y_C_S):      {y_C_S.X:.2f}")
    print(f"  Sold to R&D (z_C):     {z_C.X:.2f}")
    print(f"  Unused C (u_C):        {u_C.X:.2f}")

    # FinalAnswer is the maximum profit Z
    print(f"FinalAnswer=【{model.objVal:.4f}】")
else:
    print("No optimal solution found.")
    # In case no optimal solution, set FinalAnswer to 0 or a clear indicator
    print("FinalAnswer=【NaN】")