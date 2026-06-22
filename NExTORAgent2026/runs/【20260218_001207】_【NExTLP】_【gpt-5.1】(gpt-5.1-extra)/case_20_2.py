import gurobipy as gp
from gurobipy import GRB

# ============================
# 1. Parameters (from Parameters List)
# ============================

Food_types = ['Rice', 'Chicken', 'Beans', 'Milk', 'Vegetables']
Pack_types = ['Pack A', 'Pack B', 'Pack C', 'Pack D', 'Vegetables']

Daily_min_calories_normal = 2000
Daily_max_calories_normal = 2500
Daily_min_calories_incentive = 2500   # not used directly, encoded via y_d
Daily_max_calories_incentive = 3000   # not used directly, encoded via y_d

Daily_min_protein = 50
Daily_max_fat = 70
Daily_min_vitaminC = 100
Daily_min_vegetables_grams = 100
Daily_max_chicken_grams = 300
Daily_max_beans_grams = 400

Num_days_per_week = 7
Num_incentive_meals_per_week = 1

Weekly_min_purchase_pack_A = 3
Weekly_min_purchase_pack_B = 1
Weekly_min_purchase_pack_C = 1
Weekly_min_purchase_pack_D = 5

Daily_delivery_limit_pack_A = 2
Daily_delivery_limit_pack_B = 5
Daily_delivery_limit_pack_C = 5
Daily_delivery_limit_pack_D = 1

# Nutrition per pack (per pack) and per gram (vegetables) – MUST use given values
Pack_A_nutrition_per_pack = {
    'Calories_kcal': 190.0,
    'Protein_g': 11.5,
    'Fat_g': 7.75,
    'VitaminC_mg': 0.0,
    'Cost_yuan': 1.125
}
Pack_B_nutrition_per_pack = {
    'Calories_kcal': 170.0,
    'Protein_g': 9.5,
    'Fat_g': 6.25,
    'VitaminC_mg': 0.0,
    'Cost_yuan': 0.925
}
Pack_C_nutrition_per_pack = {
    'Calories_kcal': 106.0,
    'Protein_g': 6.2,
    'Fat_g': 3.3,
    'VitaminC_mg': 1.0,
    'Cost_yuan': 0.65
}
Pack_D_nutrition_per_pack = {
    'Calories_kcal': 30.0,
    'Protein_g': 1.5,
    'Fat_g': 2.0,
    'VitaminC_mg': 2.5,
    'Cost_yuan': 0.75
}
Vegetables_nutrition_per_gram = {
    'Calories_kcal': 0.5,
    'Protein_g': 0.02,
    'Fat_g': 0.0,
    'VitaminC_mg': 0.2,
    'Cost_yuan': 0.01
}

# Pack composition in grams (used only for chicken/beans gram limits)
Pack_composition = {
    'Pack A': {'Rice': 25, 'Chicken': 50, 'Beans': 0,  'Milk': 0},
    'Pack B': {'Rice': 25, 'Chicken': 40, 'Beans': 0,  'Milk': 0},
    'Pack C': {'Rice': 10, 'Chicken': 20, 'Beans': 20, 'Milk': 0},
    'Pack D': {'Rice': 0,  'Chicken': 0,  'Beans': 0,  'Milk': 50}
}

# ============================
# 2. Create model
# ============================

model = gp.Model("Weekly_Fat_Reducing_Energy_Pack_Optimization")

days = range(1, Num_days_per_week + 1)

# ============================
# 3. Decision variables
# ============================

# Packs per day (integer)
A = model.addVars(days, vtype=GRB.INTEGER, name="A")   # Pack A
B = model.addVars(days, vtype=GRB.INTEGER, name="B")   # Pack B
C = model.addVars(days, vtype=GRB.INTEGER, name="C")   # Pack C
D = model.addVars(days, vtype=GRB.INTEGER, name="D")   # Pack D

# Vegetables grams per day (continuous ≥ 0)
V = model.addVars(days, vtype=GRB.CONTINUOUS, lb=0.0, name="V")

# Incentive day indicators
y = model.addVars(days, vtype=GRB.BINARY, name="y")

# Auxiliary nutrition variables
Cal  = model.addVars(days, vtype=GRB.CONTINUOUS, name="Cal")
Prot = model.addVars(days, vtype=GRB.CONTINUOUS, name="Prot")
Fat  = model.addVars(days, vtype=GRB.CONTINUOUS, name="Fat")
VitC = model.addVars(days, vtype=GRB.CONTINUOUS, name="VitC")

# ============================
# 4. Nutrition definitions
# ============================

for d in days:
    # Calories
    model.addConstr(
        Cal[d] ==
        Pack_A_nutrition_per_pack['Calories_kcal'] * A[d] +
        Pack_B_nutrition_per_pack['Calories_kcal'] * B[d] +
        Pack_C_nutrition_per_pack['Calories_kcal'] * C[d] +
        Pack_D_nutrition_per_pack['Calories_kcal'] * D[d] +
        Vegetables_nutrition_per_gram['Calories_kcal'] * V[d],
        name=f"Cal_def_{d}"
    )

    # Protein
    model.addConstr(
        Prot[d] ==
        Pack_A_nutrition_per_pack['Protein_g'] * A[d] +
        Pack_B_nutrition_per_pack['Protein_g'] * B[d] +
        Pack_C_nutrition_per_pack['Protein_g'] * C[d] +
        Pack_D_nutrition_per_pack['Protein_g'] * D[d] +
        Vegetables_nutrition_per_gram['Protein_g'] * V[d],
        name=f"Prot_def_{d}"
    )

    # Fat
    model.addConstr(
        Fat[d] ==
        Pack_A_nutrition_per_pack['Fat_g'] * A[d] +
        Pack_B_nutrition_per_pack['Fat_g'] * B[d] +
        Pack_C_nutrition_per_pack['Fat_g'] * C[d] +
        Pack_D_nutrition_per_pack['Fat_g'] * D[d] +
        Vegetables_nutrition_per_gram['Fat_g'] * V[d],
        name=f"Fat_def_{d}"
    )

    # Vitamin C
    model.addConstr(
        VitC[d] ==
        Pack_A_nutrition_per_pack['VitaminC_mg'] * A[d] +
        Pack_B_nutrition_per_pack['VitaminC_mg'] * B[d] +
        Pack_C_nutrition_per_pack['VitaminC_mg'] * C[d] +
        Pack_D_nutrition_per_pack['VitaminC_mg'] * D[d] +
        Vegetables_nutrition_per_gram['VitaminC_mg'] * V[d],
        name=f"VitC_def_{d}"
    )

# ============================
# 5. Daily constraints
# ============================

for d in days:
    # Calorie bounds with incentive logic:
    #  normal day: 2000–2500
    #  incentive: 2500–3000 (= +500 both bounds)
    model.addConstr(
        Cal[d] >= Daily_min_calories_normal + 500 * y[d],
        name=f"Cal_min_{d}"
    )
    model.addConstr(
        Cal[d] <= Daily_max_calories_normal + 500 * y[d],
        name=f"Cal_max_{d}"
    )

    # Protein minimum
    model.addConstr(
        Prot[d] >= Daily_min_protein,
        name=f"Prot_min_{d}"
    )

    # Fat maximum
    model.addConstr(
        Fat[d] <= Daily_max_fat,
        name=f"Fat_max_{d}"
    )

    # Vitamin C minimum
    model.addConstr(
        VitC[d] >= Daily_min_vitaminC,
        name=f"VitC_min_{d}"
    )

    # Vegetables minimum (grams)
    model.addConstr(
        V[d] >= Daily_min_vegetables_grams,
        name=f"Veg_min_{d}"
    )

    # Chicken gram limit: 50*A + 40*B + 20*C ≤ 300
    chicken_grams = (
        Pack_composition['Pack A']['Chicken'] * A[d] +
        Pack_composition['Pack B']['Chicken'] * B[d] +
        Pack_composition['Pack C']['Chicken'] * C[d]
    )
    model.addConstr(
        chicken_grams <= Daily_max_chicken_grams,
        name=f"Chicken_max_{d}"
    )

    # Beans gram limit: 20*C ≤ 400
    beans_grams = Pack_composition['Pack C']['Beans'] * C[d]
    model.addConstr(
        beans_grams <= Daily_max_beans_grams,
        name=f"Beans_max_{d}"
    )

    # Daily delivery limit per pack
    model.addConstr(A[d] <= Daily_delivery_limit_pack_A, name=f"A_daily_limit_{d}")
    model.addConstr(B[d] <= Daily_delivery_limit_pack_B, name=f"B_daily_limit_{d}")
    model.addConstr(C[d] <= Daily_delivery_limit_pack_C, name=f"C_daily_limit_{d}")
    model.addConstr(D[d] <= Daily_delivery_limit_pack_D, name=f"D_daily_limit_{d}")

# ============================
# 6. Weekly constraints
# ============================

# Weekly minimum purchase quantities
model.addConstr(
    gp.quicksum(A[d] for d in days) >= Weekly_min_purchase_pack_A,
    name="Weekly_min_A"
)
model.addConstr(
    gp.quicksum(B[d] for d in days) >= Weekly_min_purchase_pack_B,
    name="Weekly_min_B"
)
model.addConstr(
    gp.quicksum(C[d] for d in days) >= Weekly_min_purchase_pack_C,
    name="Weekly_min_C"
)
model.addConstr(
    gp.quicksum(D[d] for d in days) >= Weekly_min_purchase_pack_D,
    name="Weekly_min_D"
)

# Exactly one incentive day
model.addConstr(
    gp.quicksum(y[d] for d in days) == Num_incentive_meals_per_week,
    name="Exactly_one_incentive_day"
)

# ============================
# 7. Objective: minimize weekly total cost
# ============================

total_cost = gp.quicksum(
    Pack_A_nutrition_per_pack['Cost_yuan'] * A[d] +
    Pack_B_nutrition_per_pack['Cost_yuan'] * B[d] +
    Pack_C_nutrition_per_pack['Cost_yuan'] * C[d] +
    Pack_D_nutrition_per_pack['Cost_yuan'] * D[d] +
    Vegetables_nutrition_per_gram['Cost_yuan'] * V[d]
    for d in days
)

model.setObjective(total_cost, GRB.MINIMIZE)

# ============================
# 8. Optimize
# ============================

model.optimize()

# ============================
# 9. Print results
# ============================

if model.status == GRB.OPTIMAL:
    print(f"Optimal total weekly cost: {model.objVal:.4f} yuan\n")

    total_A = sum(A[d].X for d in days)
    total_B = sum(B[d].X for d in days)
    total_C = sum(C[d].X for d in days)
    total_D = sum(D[d].X for d in days)
    total_V = sum(V[d].X for d in days)

    print("Weekly purchase / consumption summary:")
    print(f"  Pack A: {total_A:.0f} bags")
    print(f"  Pack B: {total_B:.0f} bags")
    print(f"  Pack C: {total_C:.0f} bags")
    print(f"  Pack D: {total_D:.0f} bags")
    print(f"  Vegetables: {total_V:.1f} grams\n")

    print("Daily allocation and nutrition:")
    for d in days:
        print(f"Day {d}:")
        print(f"  A_d = {A[d].X:.0f}, B_d = {B[d].X:.0f}, C_d = {C[d].X:.0f}, D_d = {D[d].X:.0f}")
        print(f"  V_d = {V[d].X:.1f} g, Incentive day y_d = {int(round(y[d].X))}")
        print(f"  Calories = {Cal[d].X:.1f} kcal, Protein = {Prot[d].X:.1f} g, "
              f"Fat = {Fat[d].X:.1f} g, VitC = {VitC[d].X:.1f} mg\n")

    # FinalAnswer: total weekly cost (the question asks for the calculated total weekly cost)
    print(f"FinalAnswer=【{model.objVal:.4f}】")
else:
    print(f"Model did not find an optimal solution. Status code: {model.status}")
    # If infeasible or other, still print something for FinalAnswer
    print("FinalAnswer=【NaN】")