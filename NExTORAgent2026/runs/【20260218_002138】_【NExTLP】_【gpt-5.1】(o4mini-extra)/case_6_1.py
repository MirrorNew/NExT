import gurobipy as gp
from gurobipy import GRB

# ==============================
# 1. Parameters (from Parameters List)
# ==============================
total_area = 7
block_A_area = 5
block_B_area = 2
crops_in_block_A = ['Green corn', 'Cabbage', 'Tomato']
water_limit = 5000
labor_people = 400
hours_per_person = 10
labor_limit = labor_people * hours_per_person
min_number_of_crops = 6
min_area_if_planted = 0.8

Table_1_CropData = [
    {'crop': 'Green corn',   'profit_per_ha': 10000, 'water_per_ha': 800, 'labor_per_ha': 600, 'max_requirement_ha': 2},
    {'crop': 'Cabbage',      'profit_per_ha': 12000, 'water_per_ha': 700, 'labor_per_ha': 500, 'max_requirement_ha': 3},
    {'crop': 'Tomato',       'profit_per_ha': 15000, 'water_per_ha': 900, 'labor_per_ha': 700, 'max_requirement_ha': 1.5},
    {'crop': 'Spinach',      'profit_per_ha': 8000,  'water_per_ha': 600, 'labor_per_ha': 400, 'max_requirement_ha': 1},
    {'crop': 'Mustard',      'profit_per_ha': 9000,  'water_per_ha': 650, 'labor_per_ha': 450, 'max_requirement_ha': 1.5},
    {'crop': 'Pumpkin',      'profit_per_ha': 11000, 'water_per_ha': 750, 'labor_per_ha': 550, 'max_requirement_ha': 1},
    {'crop': 'Sweet potato', 'profit_per_ha': 10000, 'water_per_ha': 700, 'labor_per_ha': 500, 'max_requirement_ha': 1}
]

# Map crop names to short keys consistent with context
crop_key_map = {
    'Green corn': 'GC',
    'Cabbage': 'Ca',
    'Tomato': 'To',
    'Spinach': 'Sp',
    'Mustard': 'Mu',
    'Pumpkin': 'Pu',
    'Sweet potato': 'Sw'
}

# Build helper dictionaries indexed by short keys
profits = {}
water = {}
labor = {}
max_req = {}

for item in Table_1_CropData:
    k = crop_key_map[item['crop']]
    profits[k] = item['profit_per_ha']
    water[k] = item['water_per_ha']
    labor[k] = item['labor_per_ha']
    max_req[k] = item['max_requirement_ha']

crops = list(profits.keys())

# ==============================
# 2. Create model
# ==============================
model = gp.Model("Nonglian_Land_Allocation")

# ==============================
# 3. Decision variables
# ==============================
# Continuous area variables (ha)
x = model.addVars(crops, name="x", lb=0.0, vtype=GRB.CONTINUOUS)

# Binary planting decision variables
y = model.addVars(crops, name="y", vtype=GRB.BINARY)

# Explicitly set individual upper bounds from max requirements
for k in crops:
    x[k].UB = max_req[k]

# ==============================
# 4. Objective function: maximize total profit
# ==============================
model.setObjective(
    gp.quicksum(profits[k] * x[k] for k in crops),
    GRB.MAXIMIZE
)

# ==============================
# 5. Constraints
# ==============================

# 5.1 Total land constraint
model.addConstr(
    gp.quicksum(x[k] for k in crops) <= total_area,
    name="TotalLand"
)

# 5.2 Block A land constraint (only GC, Ca, To can be in block A; sum of their area <= block_A_area)
model.addConstr(
    x['GC'] + x['Ca'] + x['To'] <= block_A_area,
    name="BlockA"
)

# 5.3 Water limit
model.addConstr(
    gp.quicksum(water[k] * x[k] for k in crops) <= water_limit,
    name="WaterLimit"
)

# 5.4 Labor limit
model.addConstr(
    gp.quicksum(labor[k] * x[k] for k in crops) <= labor_limit,
    name="LaborLimit"
)

# 5.5 Minimum-area-if-planted: x_i >= min_area_if_planted * y_i
for k in crops:
    model.addConstr(
        x[k] >= min_area_if_planted * y[k],
        name=f"MinAreaIfPlanted_{k}"
    )

# 5.6 Kill-zero linkage using indicator constraints (no big-M):
# If y_k == 0 then x_k <= 0; y_k in {0,1}, x_k >= 0 already
# Implement: model.addGenConstrIndicator(y_k, 0, x_k <= 0)
for k in crops:
    model.addGenConstrIndicator(
        y[k],            # binary variable
        0,               # binval
        x[k] <= 0,       # enforced when y[k] == 0
        name=f"ZeroAreaIfNotPlanted_{k}"
    )

# 5.7 Diversity: at least min_number_of_crops crops planted
model.addConstr(
    gp.quicksum(y[k] for k in crops) >= min_number_of_crops,
    name="Diversity"
)

# 5.8 Pumpkin–sweet potato balance: x_Pu - x_Sw = 0
model.addConstr(
    x['Pu'] - x['Sw'] == 0,
    name="PumpkinSweetPotatoBalance"
)

# 5.9 Green corn & pumpkin minimum: x_GC + x_Pu >= 3
model.addConstr(
    x['GC'] + x['Pu'] >= 3,
    name="GCPumpkinMin"
)

# ==============================
# 6. Solve the model
# ==============================
model.optimize()

# ==============================
# 7. Print results
# ==============================
if model.Status == GRB.OPTIMAL:
    print("Optimal solution found.")
    print(f"Maximum total profit = {model.ObjVal:.2f} USD")
    for full_name, short in crop_key_map.items():
        print(f"{full_name} area (x_{short}) = {x[short].X:.4f} ha, planted (y_{short}) = {int(y[short].X + 0.5)}")
else:
    print(f"Optimization ended with status {model.Status}")

# Final answer: maximum total profit only
the_question_answer = model.ObjVal if model.Status == GRB.OPTIMAL else float('nan')
print(f"FinalAnswer=【{the_question_answer}】")