import gurobipy as gp
from gurobipy import GRB

# ==========================
# 1. Parameters (from Parameters List)
# ==========================
crop_names = ['green_corn', 'cabbage', 'tomato', 'spinach', 'mustard', 'pumpkin', 'sweet_potato']
num_crops = 7
land_total = 7.0
land_block_A = 5.0
land_block_B = 2.0
water_total = 5000.0
labor_people = 400
labor_hours_per_person = 10.0
labor_total_hours = 4000.0
profit_per_ha = [10000.0, 12000.0, 15000.0, 8000.0, 9000.0, 11000.0, 10000.0]
water_per_ha = [800.0, 700.0, 900.0, 600.0, 650.0, 750.0, 700.0]
labor_per_ha = [600.0, 500.0, 700.0, 400.0, 450.0, 550.0, 500.0]
max_demand_ha = [2.0, 3.0, 1.5, 1.0, 1.5, 1.0, 1.0]
min_area_if_planted = 0.8
min_num_crops_planted = 6
block_A_crops_indices = [0, 1, 2]          # green_corn, cabbage, tomato
processing_requirement_min_area = 3.0
processing_crops_indices = [0, 5]          # green_corn, pumpkin
balanced_crops_indices = [5, 6]            # pumpkin, sweet_potato
model_feasible = 0
reason_infeasible = [
    'Full combination of constraints (factory area, balanced pumpkin-sweet_potato, '
    'min 0.8 ha per planted crop, at least 6 crops, and per-crop upper bounds) '
    'leads to no feasible planting plan in mixed-integer model.'
]

# ==========================
# 2. Create model
# ==========================
model = gp.Model("Nonglian_Land_Allocation")

# ==========================
# 3. Decision variables
# ==========================
# x[i] : area (ha) for crop i
x = model.addVars(num_crops, vtype=GRB.CONTINUOUS, lb=0.0, name="x")

# y[i] : 1 if crop i is planted, 0 otherwise
y = model.addVars(num_crops, vtype=GRB.BINARY, name="y")

# ==========================
# 4. Objective: Maximize total profit
# ==========================
model.setObjective(
    gp.quicksum(profit_per_ha[i] * x[i] for i in range(num_crops)),
    sense=GRB.MAXIMIZE
)

# ==========================
# 5. Constraints
# ==========================

# 5.1 Total land area
model.addConstr(
    gp.quicksum(x[i] for i in range(num_crops)) <= land_total,
    name="Total_land_area"
)

# 5.2 Block A capacity for crops that must be in block A (green_corn, cabbage, tomato)
model.addConstr(
    gp.quicksum(x[i] for i in block_A_crops_indices) <= land_block_A,
    name="Block_A_capacity_for_A_only_crops"
)

# 5.3 Water limit
model.addConstr(
    gp.quicksum(water_per_ha[i] * x[i] for i in range(num_crops)) <= water_total,
    name="Water_limit"
)

# 5.4 Labor limit
model.addConstr(
    gp.quicksum(labor_per_ha[i] * x[i] for i in range(num_crops)) <= labor_total_hours,
    name="Labor_limit"
)

# 5.5 Maximum demand per crop (plain upper bound)
for i in range(num_crops):
    model.addConstr(
        x[i] <= max_demand_ha[i],
        name=f"Max_demand_{crop_names[i]}"
    )

# 5.6 At least min_num_crops_planted crops planted
model.addConstr(
    gp.quicksum(y[i] for i in range(num_crops)) >= min_num_crops_planted,
    name="At_least_6_crops_planted"
)

# 5.7 Min area if planted: use indicator constraints (no big-M)
for i in range(num_crops):
    # If y[i] == 1 then x[i] >= min_area_if_planted
    model.addGenConstrIndicator(
        y[i], 1, x[i] >= min_area_if_planted,
        name=f"Min_area_if_planted_{crop_names[i]}"
    )
    # If y[i] == 0 then x[i] == 0  (strong linking)
    model.addGenConstrIndicator(
        y[i], 0, x[i] == 0,
        name=f"No_area_if_not_planted_{crop_names[i]}"
    )

# 5.8 Link to max demand via indicators (x[i] <= max_demand_ha[i] when planted)
# (The base x[i] <= max_demand_ha[i] is already added, but we add the explicit indicator form)
for i in range(num_crops):
    model.addGenConstrIndicator(
        y[i], 1, x[i] <= max_demand_ha[i],
        name=f"Link_max_demand_if_planted_{crop_names[i]}"
    )

# 5.9 Balance pumpkin and sweet potato areas: x_pumpkin = x_sweet_potato
pumpkin_idx = balanced_crops_indices[0]
sweet_potato_idx = balanced_crops_indices[1]
model.addConstr(
    x[pumpkin_idx] == x[sweet_potato_idx],
    name="Balance_pumpkin_sweet_potato"
)

# 5.10 Minimum total area of green corn and pumpkin for processing requirement
gcorn_idx = processing_crops_indices[0]
pumpkin_idx = processing_crops_indices[1]
model.addConstr(
    x[gcorn_idx] + x[pumpkin_idx] >= processing_requirement_min_area,
    name="Green_corn_plus_pumpkin_minimum_area"
)

# ==========================
# 6. Optimize
# ==========================
model.optimize()

# ==========================
# 7. Print results
# ==========================
if model.Status == GRB.OPTIMAL:
    print("Optimal solution found.")
    total_profit = model.ObjVal
    print(f"Maximum total profit: {total_profit:.2f}")
    for i in range(num_crops):
        xi = x[i].X
        yi = y[i].X
        print(f"Crop {crop_names[i]}: area = {xi:.4f} ha, planted_flag = {int(round(yi))}")
    final_answer = total_profit
elif model.Status in [GRB.INFEASIBLE, GRB.INF_OR_UNBD]:
    print("Model is infeasible or unbounded.")
    # Compute IIS for more detail (optional)
    try:
        model.computeIIS()
        model.write("infeasible_model.ilp")
    except gp.GurobiError:
        pass
    print("Reason (from Parameters List):")
    for r in reason_infeasible:
        print(r)
    final_answer = float('nan')
else:
    print(f"Optimization ended with status {model.Status}.")
    final_answer = float('nan')

# ==========================
# 8. Final answer output (required format)
# ==========================
print(f"FinalAnswer=【{final_answer}】")