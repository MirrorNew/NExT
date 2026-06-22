import gurobipy as gp
from gurobipy import GRB

# 1. Parameter definitions from the provided list
total_area = 7
block_A_area = 5
block_B_area = 2
crops_in_block_A = ['Green corn', 'Cabbage', 'Tomato']
water_limit = 5000
labor_people = 400
hours_per_person = 10
total_labor_hours = labor_people * hours_per_person
min_number_of_crops = 6
min_area_if_planted = 0.8

# Data table from parameters
crop_data = [
    {'crop': 'Green corn', 'profit_per_ha': 10000, 'water_per_ha': 800, 'labor_per_ha': 600, 'max_requirement_ha': 2},
    {'crop': 'Cabbage', 'profit_per_ha': 12000, 'water_per_ha': 700, 'labor_per_ha': 500, 'max_requirement_ha': 3},
    {'crop': 'Tomato', 'profit_per_ha': 15000, 'water_per_ha': 900, 'labor_per_ha': 700, 'max_requirement_ha': 1.5},
    {'crop': 'Spinach', 'profit_per_ha': 8000, 'water_per_ha': 600, 'labor_per_ha': 400, 'max_requirement_ha': 1},
    {'crop': 'Mustard', 'profit_per_ha': 9000, 'water_per_ha': 650, 'labor_per_ha': 450, 'max_requirement_ha': 1.5},
    {'crop': 'Pumpkin', 'profit_per_ha': 11000, 'water_per_ha': 750, 'labor_per_ha': 550, 'max_requirement_ha': 1},
    {'crop': 'Sweet potato', 'profit_per_ha': 10000, 'water_per_ha': 700, 'labor_per_ha': 500, 'max_requirement_ha': 1}
]

# 2. Create the model
model = gp.Model("CropOptimization")

# 3. Decision variables
crops = [d['crop'] for d in crop_data]
# x[i]: planting area of each crop (hectares)
x = {d['crop']: model.addVar(lb=0, ub=d['max_requirement_ha'], name=f"x_{d['crop']}") for d in crop_data}
# y[i]: binary variable, 1 if crop i is planted, 0 otherwise
y = {d['crop']: model.addVar(vtype=GRB.BINARY, name=f"y_{d['crop']}") for d in crop_data}

# 4. Objective function: Maximize total profit
model.setObjective(gp.quicksum(d['profit_per_ha'] * x[d['crop']] for d in crop_data), GRB.MAXIMIZE)

# 5. Constraints
# Total available farmland constraint
model.addConstr(gp.quicksum(x[c] for c in crops) <= total_area, "Total_Farmland")

# Block A land constraint: Green corn, cabbage, and tomatoes can only be planted in block A
# This means their total area must not exceed the capacity of Block A
model.addConstr(gp.quicksum(x[c] for c in crops_in_block_A) <= block_A_area, "Block_A_Farmland")

# Irrigation water source constraint
model.addConstr(gp.quicksum(d['water_per_ha'] * x[d['crop']] for d in crop_data) <= water_limit, "Water_Limit")

# Total labor force constraint
model.addConstr(gp.quicksum(d['labor_per_ha'] * x[d['crop']] for d in crop_data) <= total_labor_hours, "Labor_Limit")

# Diversity requirement: Plant at least 6 crops
model.addConstr(gp.quicksum(y[c] for c in crops) >= min_number_of_crops, "Min_Crops_Diversity")

# If-planted conditions for each crop (minimum area if planted)
for d in crop_data:
    c = d['crop']
    # If crop is planted (y=1), its area must be at least 0.8 hectares
    model.addGenConstrIndicator(y[c], 1, x[c] >= min_area_if_planted)
    # If crop is not planted (y=0), its area must be 0
    model.addGenConstrIndicator(y[c], 0, x[c] <= 0)

# Balance constraint: Pumpkin and Sweet potato areas must be equal
model.addConstr(x['Pumpkin'] == x['Sweet potato'], "Pumpkin_SweetPotato_Balance")

# Factory supply constraint: Total area of Green corn and Pumpkin must be at least 3 hectares
model.addConstr(x['Green corn'] + x['Pumpkin'] >= 3, "Factory_Supply_Requirement")

# 6. Solve the model
model.optimize()

# 7. Print the result
if model.status == GRB.OPTIMAL:
    # Get the maximum profit
    max_profit = model.objVal
    print(f"FinalAnswer=【{max_profit}】")