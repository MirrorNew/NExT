import gurobipy as gp
from gurobipy import GRB

# =============================
# 1. Parameters (from Parameters List)
# =============================
motivational_meal_days_min = 1
calorie_min = 2000
calorie_max = 2500
protein_min = 50
vitamin_C_min = 100
fat_max = 70
vegetables_min = 100
chicken_max = 300
beans_max = 400
incentive_meal_days = 1
incentive_day_cal_min = 2500   # not used directly; encoded via y_j and 2000+500*y_j
incentive_day_cal_max = 3000   # not used directly; encoded via y_j and 2500+500*y_j

table_1_limits = {
    'minimum_weekly_purchase_quantity': {
        'Pack A': 3,
        'Pack B': 1,
        'Pack C': 1,
        'Pack D': 5,
        'Vegetables': None
    },
    'daily_delivery_limit': {
        'Pack A': 2,
        'Pack B': 5,
        'Pack C': 5,
        'Pack D': 1,
        'Vegetables': None
    }
}

table_2_pack_composition = {
    'Pack A': {'Rice': 25, 'Chicken': 50, 'Beans': 0, 'Milk': 0},
    'Pack B': {'Rice': 25, 'Chicken': 40, 'Beans': 0, 'Milk': 0},
    'Pack C': {'Rice': 10, 'Chicken': 20, 'Beans': 20, 'Milk': 0},
    'Pack D': {'Rice': 0,  'Chicken': 0,  'Beans': 0,  'Milk': 50},
    'Vegetables': {'Rice': None, 'Chicken': None, 'Beans': None, 'Milk': None}
}

table_3_nutrient_and_cost = {
    'Rice':       {'Calories': 360, 'Protein': 6,  'Fat': 1, 'Vitamin C': 0,  'Cost': 0.5},
    'Chicken':    {'Calories': 200, 'Protein': 20, 'Fat': 15,'Vitamin C': 0,  'Cost': 2.0},
    'Beans':      {'Calories': 150, 'Protein': 8,  'Fat': 1, 'Vitamin C': 5,  'Cost': 1.0},
    'Milk':       {'Calories': 60,  'Protein': 3,  'Fat': 4, 'Vitamin C': 5,  'Cost': 1.5},
    'Vegetables': {'Calories': 50,  'Protein': 2,  'Fat': 0, 'Vitamin C': 20, 'Cost': 1.0}
}

# =============================
# 2. Derived coefficients (as given in the validated model)
# =============================
# These coefficients are taken directly from the validated mathematical model:
# Calories per pack and per gram of vegetables:
cal_per_A = 190
cal_per_B = 170
cal_per_C = 106
cal_per_D = 30
cal_per_v = 0.5  # per gram of vegetables

# Protein per pack and per gram of vegetables:
prot_per_A = 11.5
prot_per_B = 9.5
prot_per_C = 6.2
prot_per_D = 1.5
prot_per_v = 0.02

# Vitamin C per pack and per gram of vegetables:
vitC_per_A = 0.0   # not used in constraint; only C, D, and vegetables appear
vitC_per_B = 0.0
vitC_per_C = 1.0
vitC_per_D = 2.5
vitC_per_v = 0.2

# Fat per pack:
fat_per_A = 7.75
fat_per_B = 6.25
fat_per_C = 3.3
fat_per_D = 2.0

# Cost per pack and per gram of vegetables (from objective expression):
cost_per_A = 1.125
cost_per_B = 0.925
cost_per_C = 0.650
cost_per_D = 0.750
cost_per_v = 0.01  # per gram of vegetables

# Chicken and beans grams per pack:
chicken_per_A = table_2_pack_composition['Pack A']['Chicken']
chicken_per_B = table_2_pack_composition['Pack B']['Chicken']
chicken_per_C = table_2_pack_composition['Pack C']['Chicken']

beans_per_C = table_2_pack_composition['Pack C']['Beans']

# Weekly minimum purchase (packs consumed in the week):
weekly_min_A = table_1_limits['minimum_weekly_purchase_quantity']['Pack A']
weekly_min_B = table_1_limits['minimum_weekly_purchase_quantity']['Pack B']
weekly_min_C = table_1_limits['minimum_weekly_purchase_quantity']['Pack C']
weekly_min_D = table_1_limits['minimum_weekly_purchase_quantity']['Pack D']

# Daily delivery upper limits:
daily_max_A = table_1_limits['daily_delivery_limit']['Pack A']
daily_max_B = table_1_limits['daily_delivery_limit']['Pack B']
daily_max_C = table_1_limits['daily_delivery_limit']['Pack C']
daily_max_D = table_1_limits['daily_delivery_limit']['Pack D']

# =============================
# 3. Create model
# =============================
model = gp.Model("Weekly_Fat_Reducing_Energy_Package")

days = range(7)  # 0..6 representing 7 days

# =============================
# 4. Decision variables
# =============================
# Integer packs per day
xA = model.addVars(days, vtype=GRB.INTEGER, name="xA")
xB = model.addVars(days, vtype=GRB.INTEGER, name="xB")
xC = model.addVars(days, vtype=GRB.INTEGER, name="xC")
xD = model.addVars(days, vtype=GRB.INTEGER, name="xD")

# Vegetables grams per day (continuous)
v = model.addVars(days, vtype=GRB.CONTINUOUS, lb=0.0, name="v")

# Incentive-day indicator (binary)
y = model.addVars(days, vtype=GRB.BINARY, name="y")

# =============================
# 5. Objective function
# =============================
model.setObjective(
    gp.quicksum(
        cost_per_A * xA[j] +
        cost_per_B * xB[j] +
        cost_per_C * xC[j] +
        cost_per_D * xD[j] +
        cost_per_v * v[j]
        for j in days
    ),
    GRB.MINIMIZE
)

# =============================
# 6. Constraints
# =============================

# 6.1 Calorie bounds with incentive shift
for j in days:
    model.addConstr(
        cal_per_A * xA[j] +
        cal_per_B * xB[j] +
        cal_per_C * xC[j] +
        cal_per_D * xD[j] +
        cal_per_v * v[j]
        >= calorie_min + 500 * y[j],
        name=f"Calorie_LB_day{j+1}"
    )

    model.addConstr(
        cal_per_A * xA[j] +
        cal_per_B * xB[j] +
        cal_per_C * xC[j] +
        cal_per_D * xD[j] +
        cal_per_v * v[j]
        <= calorie_max + 500 * y[j],
        name=f"Calorie_UB_day{j+1}"
    )

# 6.2 Protein requirement
for j in days:
    model.addConstr(
        prot_per_A * xA[j] +
        prot_per_B * xB[j] +
        prot_per_C * xC[j] +
        prot_per_D * xD[j] +
        prot_per_v * v[j]
        >= protein_min,
        name=f"Protein_day{j+1}"
    )

# 6.3 Vitamin C requirement
for j in days:
    model.addConstr(
        vitC_per_C * xC[j] +
        vitC_per_D * xD[j] +
        vitC_per_v * v[j]
        >= vitamin_C_min,
        name=f"VitaminC_day{j+1}"
    )

# 6.4 Fat limit
for j in days:
    model.addConstr(
        fat_per_A * xA[j] +
        fat_per_B * xB[j] +
        fat_per_C * xC[j] +
        fat_per_D * xD[j]
        <= fat_max,
        name=f"Fat_day{j+1}"
    )

# 6.5 Vegetables minimum per day
for j in days:
    model.addConstr(
        v[j] >= vegetables_min,
        name=f"Vegetable_min_day{j+1}"
    )

# 6.6 Chicken maximum per day
for j in days:
    model.addConstr(
        chicken_per_A * xA[j] +
        chicken_per_B * xB[j] +
        chicken_per_C * xC[j]
        <= chicken_max,
        name=f"Chicken_max_day{j+1}"
    )

# 6.7 Beans maximum per day
for j in days:
    model.addConstr(
        beans_per_C * xC[j]
        <= beans_max,
        name=f"Beans_max_day{j+1}"
    )

# 6.8 Exactly one incentive meal day
model.addConstr(
    gp.quicksum(y[j] for j in days) == incentive_meal_days,
    name="Single_incentive_day"
)

# 6.9 Daily pack-use limits
for j in days:
    model.addConstr(xA[j] <= daily_max_A, name=f"Daily_max_A_day{j+1}")
    model.addConstr(xB[j] <= daily_max_B, name=f"Daily_max_B_day{j+1}")
    model.addConstr(xC[j] <= daily_max_C, name=f"Daily_max_C_day{j+1}")
    model.addConstr(xD[j] <= daily_max_D, name=f"Daily_max_D_day{j+1}")

# 6.10 Weekly minimum purchase quantities
model.addConstr(
    gp.quicksum(xA[j] for j in days) >= weekly_min_A,
    name="Weekly_min_A"
)
model.addConstr(
    gp.quicksum(xB[j] for j in days) >= weekly_min_B,
    name="Weekly_min_B"
)
model.addConstr(
    gp.quicksum(xC[j] for j in days) >= weekly_min_C,
    name="Weekly_min_C"
)
model.addConstr(
    gp.quicksum(xD[j] for j in days) >= weekly_min_D,
    name="Weekly_min_D"
)

# No indicator-style if-then constraints beyond linear expressions are needed,
# so addGenConstrIndicator is not required here.

# =============================
# 7. Solve model
# =============================
model.optimize()

if model.Status == GRB.OPTIMAL:
    total_weekly_cost = model.ObjVal

    print("Optimal solution found.")
    print(f"Total weekly cost: {total_weekly_cost:.4f} yuan\n")

    # Packs per day
    for j in days:
        print(f"Day {j+1}:")
        print(f"  Pack A: {xA[j].X:.0f} bags")
        print(f"  Pack B: {xB[j].X:.0f} bags")
        print(f"  Pack C: {xC[j].X:.0f} bags")
        print(f"  Pack D: {xD[j].X:.0f} bags")
        print(f"  Vegetables: {v[j].X:.2f} g")
        print(f"  Incentive meal day indicator y[{j+1}]: {y[j].X:.0f}")
        print("")

    # Weekly totals
    total_A = sum(xA[j].X for j in days)
    total_B = sum(xB[j].X for j in days)
    total_C = sum(xC[j].X for j in days)
    total_D = sum(xD[j].X for j in days)
    total_veg = sum(v[j].X for j in days)

    print("Weekly totals:")
    print(f"  Pack A total: {total_A:.0f} bags")
    print(f"  Pack B total: {total_B:.0f} bags")
    print(f"  Pack C total: {total_C:.0f} bags")
    print(f"  Pack D total: {total_D:.0f} bags")
    print(f"  Vegetables total: {total_veg:.2f} g")

    # Final answer is the total weekly cost
    print(f"FinalAnswer=【{total_weekly_cost:.4f}】")
else:
    print("No optimal solution found.")
    print(f"FinalAnswer=【NaN】")