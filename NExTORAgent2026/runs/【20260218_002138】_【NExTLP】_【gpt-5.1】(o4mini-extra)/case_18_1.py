import gurobipy as gp
from gurobipy import GRB

# ==========================
# 1. Define all parameters
# ==========================

# Parameters List (given)
Total_Arable_Land_Area = 100
Sum_of_Areas_of_All_Crops = 100
Min_Planting_Area_Per_Crop = 10
Min_Planting_Area_Wheat = 20
Max_Planting_Area_Cotton = 30
Max_Planting_Area_Corn_and_Cotton = 80
Water_Consumption_Threshold_Ratio = 0.8
Labor_Per_Unit_Irrigation_Water = 2
Labor_Per_Additional_Fertilizer_Unit = 2
Max_Additional_Fertilizer = 20
Labor_Capacity_Per_Worker = 2
Total_Number_of_Workers = 500
Table_1_Crop_Data = {
    'Wheat':  [300, 10, 5, 2],  # [profit, labor, water, fert]
    'Corn':   [400, 8, 7, 3],
    'Soybean':[250, 5, 4, 1],
    'Cotton': [500, 12, 9, 4]
}
Table_2_Resource_Supply = {
    'Workers': 500,
    'Maximum limit of irrigation water resources': 500,
    'Available resources of fertilizer resources': 150,
    'Cultivated land area': 100
}

# Derived parameters (using only provided values)
max_water = Table_2_Resource_Supply['Maximum limit of irrigation water resources']
max_fertilizer_available = Table_2_Resource_Supply['Available resources of fertilizer resources']
water_threshold = Water_Consumption_Threshold_Ratio * max_water  # 0.8 * 500 = 400
max_fertilizer_total = max_fertilizer_available + Max_Additional_Fertilizer  # 150 + 20 = 170

# Crop parameter unpacking for readability
profit_w, labor_w, water_w, fert_w = Table_1_Crop_Data['Wheat']
profit_c, labor_c, water_c, fert_c = Table_1_Crop_Data['Corn']
profit_s, labor_s, water_s, fert_s = Table_1_Crop_Data['Soybean']
profit_co, labor_co, water_co, fert_co = Table_1_Crop_Data['Cotton']

# =================================
# 2. Create Gurobi model
# =================================
model = gp.Model("Agricultural_Resource_Allocation")

# =================================
# 3. Decision variables
# =================================

# Planting areas (hectares) - continuous
x_w  = model.addVar(lb=0, ub=Total_Arable_Land_Area, vtype=GRB.CONTINUOUS, name="x_w")   # wheat
x_c  = model.addVar(lb=0, ub=Total_Arable_Land_Area, vtype=GRB.CONTINUOUS, name="x_c")   # corn
x_s  = model.addVar(lb=0, ub=Total_Arable_Land_Area, vtype=GRB.CONTINUOUS, name="x_s")   # soybean
x_co = model.addVar(lb=0, ub=Total_Arable_Land_Area, vtype=GRB.CONTINUOUS, name="x_co")  # cotton

# Workers assigned to each crop (integer, persons)
w_w  = model.addVar(lb=0, ub=Total_Number_of_Workers, vtype=GRB.INTEGER, name="w_w")
w_c  = model.addVar(lb=0, ub=Total_Number_of_Workers, vtype=GRB.INTEGER, name="w_c")
w_s  = model.addVar(lb=0, ub=Total_Number_of_Workers, vtype=GRB.INTEGER, name="w_s")
w_co = model.addVar(lb=0, ub=Total_Number_of_Workers, vtype=GRB.INTEGER, name="w_co")

# Total irrigation water used (thousand m3)
W = model.addVar(lb=0, ub=max_water, vtype=GRB.CONTINUOUS, name="W")

# Total fertilizer used (tons)
F = model.addVar(lb=0, ub=max_fertilizer_total, vtype=GRB.CONTINUOUS, name="F")

# Additional labor for water transport (man-hours)
lab_w = model.addVar(lb=0, ub=(max_water - water_threshold) * Labor_Per_Unit_Irrigation_Water,
                     vtype=GRB.CONTINUOUS, name="lab_w")

# Additional labor for fertilizer purchase (man-hours)
lab_f = model.addVar(lb=0, ub=Max_Additional_Fertilizer * Labor_Per_Additional_Fertilizer_Unit,
                     vtype=GRB.CONTINUOUS, name="lab_f")

# =================================
# 4. Objective function: Max profit
# =================================
model.setObjective(
    profit_w  * x_w  +
    profit_c  * x_c  +
    profit_s  * x_s  +
    profit_co * x_co,
    GRB.MAXIMIZE
)

# =================================
# 5. Constraints
# =================================

# (1) Total land usage (all land planted)
model.addConstr(
    x_w + x_c + x_s + x_co == Sum_of_Areas_of_All_Crops,
    name="Total_land_usage"
)

# (2) Minimum area for each crop
model.addConstr(x_w  >= Min_Planting_Area_Wheat,        name="Min_area_wheat")
model.addConstr(x_c  >= Min_Planting_Area_Per_Crop,     name="Min_area_corn")
model.addConstr(x_s  >= Min_Planting_Area_Per_Crop,     name="Min_area_soybean")
model.addConstr(x_co >= Min_Planting_Area_Per_Crop,     name="Min_area_cotton")

# (3) Maximum cotton area
model.addConstr(x_co <= Max_Planting_Area_Cotton, name="Max_cotton_area")

# (4) Corn + Cotton combined limit
model.addConstr(
    x_c + x_co <= Max_Planting_Area_Corn_and_Cotton,
    name="Corn_Cotton_limit"
)

# (5) Irrigation water definition
model.addConstr(
    W == water_w * x_w + water_c * x_c + water_s * x_s + water_co * x_co,
    name="Water_definition"
)

# (6) Water usage limit
model.addConstr(W <= max_water, name="Water_limit")

# (7) Water transport additional labor (linearization)
# lab_w >= 2*(W - 400)
model.addConstr(
    lab_w >= Labor_Per_Unit_Irrigation_Water * (W - water_threshold),
    name="Water_transport_labor_ge"
)
# lab_w >= 0 (already via lb=0, but we add constraint as explicitly given)
model.addConstr(
    lab_w >= 0,
    name="Water_transport_labor_nonneg"
)

# (8) Fertilizer usage definition
model.addConstr(
    F == fert_w * x_w + fert_c * x_c + fert_s * x_s + fert_co * x_co,
    name="Fertilizer_definition"
)

# (9) Fertilizer maximum consumption (total cannot exceed 170)
model.addConstr(F <= max_fertilizer_total, name="Fertilizer_max_consumption")

# (10) Fertilizer transport additional labor (linearization)
# lab_f >= 2*(F - 150)
model.addConstr(
    lab_f >= Labor_Per_Additional_Fertilizer_Unit * (F - max_fertilizer_available),
    name="Fertilizer_transport_labor_ge"
)
# lab_f >= 0 (already via lb=0, but we add constraint as explicitly given)
model.addConstr(
    lab_f >= 0,
    name="Fertilizer_transport_labor_nonneg"
)

# (11) Planting labor requirement
model.addConstr(
    Labor_Capacity_Per_Worker * (w_w + w_c + w_s + w_co) >=
    labor_w * x_w + labor_c * x_c + labor_s * x_s + labor_co * x_co,
    name="Planting_labor_requirement"
)

# (12) Total workers availability
model.addConstr(
    w_w + w_c + w_s + w_co <= Total_Number_of_Workers,
    name="Total_workers_limit"
)

# =================================
# 6. Optimize the model
# =================================
model.optimize()

# =================================
# 7. Print solution
# =================================
if model.Status == GRB.OPTIMAL:
    print("Optimal objective (maximum profit):", model.ObjVal)
    print(f"x_w  (wheat area)   = {x_w.X:.4f} ha")
    print(f"x_c  (corn area)    = {x_c.X:.4f} ha")
    print(f"x_s  (soybean area) = {x_s.X:.4f} ha")
    print(f"x_co (cotton area)  = {x_co.X:.4f} ha")
    print(f"w_w  (wheat workers)   = {w_w.X:.0f}")
    print(f"w_c  (corn workers)    = {w_c.X:.0f}")
    print(f"w_s  (soybean workers) = {w_s.X:.0f}")
    print(f"w_co (cotton workers)  = {w_co.X:.0f}")
    print(f"W (total water)        = {W.X:.4f}")
    print(f"F (total fertilizer)   = {F.X:.4f}")
    print(f"lab_w (extra water labor)      = {lab_w.X:.4f}")
    print(f"lab_f (extra fertilizer labor) = {lab_f.X:.4f}")
else:
    print("No optimal solution found. Status code:", model.Status)

# =================================
# 8. Output final answer in required format
#     The question asks: "Give the maximum profit value."
# =================================
final_answer = model.ObjVal if model.Status == GRB.OPTIMAL else None
print(f"FinalAnswer=【{final_answer}】")