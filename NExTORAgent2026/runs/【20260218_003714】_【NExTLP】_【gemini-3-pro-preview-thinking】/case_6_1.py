import gurobipy as gp
from gurobipy import GRB

# 1. Define Data
# General Parameters
total_area = 7
block_A_area = 5
water_limit = 5000
labor_limit = 400 * 10  # 4000
min_number_of_crops = 6
min_area_if_planted = 0.8

# Crops Data
crops_data = [
    {'crop': 'Green corn', 'profit': 10000, 'water': 800, 'labor': 600, 'max_ha': 2},
    {'crop': 'Cabbage', 'profit': 12000, 'water': 700, 'labor': 500, 'max_ha': 3},
    {'crop': 'Tomato', 'profit': 15000, 'water': 900, 'labor': 700, 'max_ha': 1.5},
    {'crop': 'Spinach', 'profit': 8000, 'water': 600, 'labor': 400, 'max_ha': 1},
    {'crop': 'Mustard', 'profit': 9000, 'water': 650, 'labor': 450, 'max_ha': 1.5},
    {'crop': 'Pumpkin', 'profit': 11000, 'water': 750, 'labor': 550, 'max_ha': 1},
    {'crop': 'Sweet potato', 'profit': 10000, 'water': 700, 'labor': 500, 'max_ha': 1}
]

crops = [d['crop'] for d in crops_data]
profit = {d['crop']: d['profit'] for d in crops_data}
water = {d['crop']: d['water'] for d in crops_data}
labor = {d['crop']: d['labor'] for d in crops_data}
max_req = {d['crop']: d['max_ha'] for d in crops_data}

# Block A crops restriction
block_A_crops = ['Green corn', 'Cabbage', 'Tomato']

# 2. Create Model
model = gp.Model("Agricultural_Optimization")

# 3. Decision Variables
# Continuous variables for planting area
x = model.addVars(crops, lb=0.0, name="x")

# Binary variables for planting decision
y = model.addVars(crops, vtype=GRB.BINARY, name="y")

# 4. Objective Function
# Maximize total profit
model.setObjective(gp.quicksum(profit[i] * x[i] for i in crops), GRB.MAXIMIZE)

# 5. Constraints

# 5.1 Total land constraint
model.addConstr(gp.quicksum(x[i] for i in crops) <= total_area, "Total_Land")

# 5.2 Block A land constraint
model.addConstr(gp.quicksum(x[i] for i in block_A_crops) <= block_A_area, "Block_A_Land")

# 5.3 Water limit
model.addConstr(gp.quicksum(water[i] * x[i] for i in crops) <= water_limit, "Water_Limit")

# 5.4 Labor limit
model.addConstr(gp.quicksum(labor[i] * x[i] for i in crops) <= labor_limit, "Labor_Limit")

# 5.5 Maximum demand per crop
# Strict upper bound
for i in crops:
    model.addConstr(x[i] <= max_req[i], f"Max_Demand_{i}")

# 5.6 Minimum area if planted (using Indicator Constraints)
# If y[i] = 1, then x[i] >= 0.8
# If y[i] = 0, then x[i] <= 0 (effectively 0 due to lb=0)
for i in crops:
    model.addGenConstrIndicator(y[i], 1, x[i] >= min_area_if_planted, name=f"Min_Area_Active_{i}")
    model.addGenConstrIndicator(y[i], 0, x[i] <= 0.0, name=f"Zero_Area_Inactive_{i}")

# 5.7 Diversity count
model.addConstr(gp.quicksum(y[i] for i in crops) >= min_number_of_crops, "Diversity_Count")

# 5.8 Pumpkin–Sweet potato balance
model.addConstr(x['Pumpkin'] == x['Sweet potato'], "Pumpkin_SweetPotato_Balance")

# 5.9 Green corn & pumpkin minimum total area
model.addConstr(x['Green corn'] + x['Pumpkin'] >= 3, "GreenCorn_Pumpkin_Min")

# 6. Solve and Print
model.optimize()

if model.status == GRB.OPTIMAL:
    print("Optimal Solution Found:")
    for i in crops:
        if x[i].x > 0:
            print(f"  {i}: {x[i].x:.2f} ha (Planted: {y[i].x})")
    
    final_profit = model.objVal
    print(f"Total Profit: {final_profit}")
    print(f"FinalAnswer=【{final_profit}】")
else:
    print("No optimal solution found.")