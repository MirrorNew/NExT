import gurobipy as gp
from gurobipy import GRB

# ================ 2. Define Parameters ================
total_area = 7
block_A_area = 5
block_B_area = 2  # Note: not directly used in constraints but provided
water_limit = 5000
labor_hours = 400 * 10  # 400 people * 10 hours each = 4000 hours
min_crops = 6
min_area_if_planted = 0.8

# Table 1 data
crop_data = [
    {'crop': 'Green corn', 'profit': 10000, 'water': 800, 'labor': 600, 'max': 2},
    {'crop': 'Cabbage', 'profit': 12000, 'water': 700, 'labor': 500, 'max': 3},
    {'crop': 'Tomato', 'profit': 15000, 'water': 900, 'labor': 700, 'max': 1.5},
    {'crop': 'Spinach', 'profit': 8000, 'water': 600, 'labor': 400, 'max': 1},
    {'crop': 'Mustard', 'profit': 9000, 'water': 650, 'labor': 450, 'max': 1.5},
    {'crop': 'Pumpkin', 'profit': 11000, 'water': 750, 'labor': 550, 'max': 1},
    {'crop': 'Sweet potato', 'profit': 10000, 'water': 700, 'labor': 500, 'max': 1}
]

crop_names = ['GC', 'Ca', 'To', 'Sp', 'Mu', 'Pu', 'Sw']

# ================ 3. Create Model ================
model = gp.Model("Nonglian_Land_Allocation")

# ================ 4. Create Decision Variables ================
x = {}
y = {}
for i, crop in enumerate(crop_names):
    x[crop] = model.addVar(lb=0, ub=crop_data[i]['max'], name=f'x_{crop}')
    y[crop] = model.addVar(vtype=GRB.BINARY, name=f'y_{crop}')

# ================ 5. Set Objective ================
objective = gp.quicksum(crop_data[i]['profit'] * x[crop] for i, crop in enumerate(crop_names))
model.setObjective(objective, GRB.MAXIMIZE)

# ================ 6. Add Constraints ================
# Total land constraint
model.addConstr(x['GC'] + x['Ca'] + x['To'] + x['Sp'] + x['Mu'] + x['Pu'] + x['Sw'] <= total_area, "Total_land")

# Block A land constraint (only GC, Ca, To can be in Block A)
model.addConstr(x['GC'] + x['Ca'] + x['To'] <= block_A_area, "Block_A_land")

# Water limit
water_cons = gp.quicksum(crop_data[i]['water'] * x[crop] for i, crop in enumerate(crop_names))
model.addConstr(water_cons <= water_limit, "Water_limit")

# Labor limit
labor_cons = gp.quicksum(crop_data[i]['labor'] * x[crop] for i, crop in enumerate(crop_names))
model.addConstr(labor_cons <= labor_hours, "Labor_limit")

# Minimum-area-if-planted and kill-zero linkage constraints
for i, crop in enumerate(crop_names):
    model.addConstr(x[crop] >= min_area_if_planted * y[crop], f"Min_area_{crop}")
    model.addConstr(x[crop] <= crop_data[i]['max'] * y[crop], f"Max_area_binary_{crop}")

# Diversity count
model.addConstr(y['GC'] + y['Ca'] + y['To'] + y['Sp'] + y['Mu'] + y['Pu'] + y['Sw'] >= min_crops, "Diversity")

# Pumpkin–sweet potato balance
model.addConstr(x['Pu'] - x['Sw'] == 0, "Pumpkin_SweetPotato_balance")

# Green corn & pumpkin minimum
model.addConstr(x['GC'] + x['Pu'] >= 3, "GC_Pumpkin_minimum")

# ================ 7. Solve and Print Results ================
model.optimize()

if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"Maximum Total Profit: ${model.ObjVal:.2f}")
    print("\nPlanting areas (ha):")
    for crop in crop_names:
        if y[crop].X > 0.5:
            print(f"  {crop}: {x[crop].X:.3f} ha")
        else:
            print(f"  {crop}: 0 ha (not planted)")
    print("\nResource usage:")
    total_water = sum(crop_data[i]['water'] * x[crop].X for i, crop in enumerate(crop_names))
    total_labor = sum(crop_data[i]['labor'] * x[crop].X for i, crop in enumerate(crop_names))
    total_land = sum(x[crop].X for crop in crop_names)
    print(f"  Water used: {total_water:.1f} m³ / {water_limit} m³")
    print(f"  Labor used: {total_labor:.1f} hours / {labor_hours} hours")
    print(f"  Total land used: {total_land:.2f} ha / {total_area} ha")
    print(f"  Crops planted: {sum(y[crop].X for crop in crop_names):.0f}")
    
    # Answer to the question: maximum total profit
    print(f"\nFinalAnswer=【{model.ObjVal:.2f}】")
else:
    print("No optimal solution found.")
    print("FinalAnswer=【No feasible solution】")