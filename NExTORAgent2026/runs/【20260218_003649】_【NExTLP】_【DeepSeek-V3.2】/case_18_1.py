import gurobipy as gp
from gurobipy import GRB

# ========== Parameters ==========
# Table 1: Crop Data (profit, labor_demand, water_demand, fertilizer_demand)
crop_data = {
    'Wheat': [300, 10, 5, 2],
    'Corn': [400, 8, 7, 3],
    'Soybean': [250, 5, 4, 1],
    'Cotton': [500, 12, 9, 4]
}

# Table 2: Resource Supply
resource_supply = {
    'Workers': 500,
    'Maximum limit of irrigation water resources': 500,
    'Available resources of fertilizer resources': 150,
    'Cultivated land area': 100
}

# Other parameters from the list
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

# Derived parameters
water_threshold = Water_Consumption_Threshold_Ratio * 500  # 80% of 500

# ========== Create Model ==========
model = gp.Model("Agricultural_Production_Planning")

# ========== Decision Variables ==========
# Planting areas (continuous)
x_w = model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS, name="x_w")
x_c = model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS, name="x_c")
x_s = model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS, name="x_s")
x_co = model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS, name="x_co")

# Workers assigned to each crop (integer)
w_w = model.addVar(lb=0, ub=500, vtype=GRB.INTEGER, name="w_w")
w_c = model.addVar(lb=0, ub=500, vtype=GRB.INTEGER, name="w_c")
w_s = model.addVar(lb=0, ub=500, vtype=GRB.INTEGER, name="w_s")
w_co = model.addVar(lb=0, ub=500, vtype=GRB.INTEGER, name="w_co")

# Total irrigation water used (continuous)
W = model.addVar(lb=0, ub=500, vtype=GRB.CONTINUOUS, name="W")

# Total fertilizer used (continuous)
F = model.addVar(lb=0, ub=170, vtype=GRB.CONTINUOUS, name="F")

# Additional labor for water transport (continuous)
lab_w = model.addVar(lb=0, ub=200, vtype=GRB.CONTINUOUS, name="lab_w")

# Additional labor for fertilizer purchase (continuous)
lab_f = model.addVar(lb=0, ub=40, vtype=GRB.CONTINUOUS, name="lab_f")

# ========== Objective Function ==========
profit_expr = (300 * x_w + 400 * x_c + 250 * x_s + 500 * x_co)
model.setObjective(profit_expr, GRB.MAXIMIZE)

# ========== Constraints ==========
# 1. Total land usage
model.addConstr(x_w + x_c + x_s + x_co == Sum_of_Areas_of_All_Crops, name="total_land")

# 2. Minimum area for each crop
model.addConstr(x_w >= Min_Planting_Area_Wheat, name="min_wheat_area")
model.addConstr(x_c >= Min_Planting_Area_Per_Crop, name="min_corn_area")
model.addConstr(x_s >= Min_Planting_Area_Per_Crop, name="min_soybean_area")
model.addConstr(x_co >= Min_Planting_Area_Per_Crop, name="min_cotton_area")

# 3. Maximum cotton area
model.addConstr(x_co <= Max_Planting_Area_Cotton, name="max_cotton_area")

# 4. Corn + Cotton limit
model.addConstr(x_c + x_co <= Max_Planting_Area_Corn_and_Cotton, name="corn_cotton_limit")

# 5. Irrigation water definition
model.addConstr(W == 5 * x_w + 7 * x_c + 4 * x_s + 9 * x_co, name="water_definition")

# 6. Water usage limit (implicitly enforced by W's upper bound)

# 7. Water transport labor (linearization)
model.addConstr(lab_w >= Labor_Per_Unit_Irrigation_Water * (W - water_threshold), name="water_labor_1")
model.addConstr(lab_w >= 0, name="water_labor_nonneg")

# 8. Fertilizer usage definition
model.addConstr(F == 2 * x_w + 3 * x_c + 1 * x_s + 4 * x_co, name="fertilizer_definition")

# 9. Fertilizer maximum consumption (implicitly enforced by F's upper bound)

# 10. Fertilizer transport labor (linearization)
model.addConstr(lab_f >= Labor_Per_Additional_Fertilizer_Unit * (F - 150), name="fertilizer_labor_1")
model.addConstr(lab_f >= 0, name="fertilizer_labor_nonneg")

# 11. Planting labor requirement
model.addConstr(
    Labor_Capacity_Per_Worker * (w_w + w_c + w_s + w_co) >= 
    10 * x_w + 8 * x_c + 5 * x_s + 12 * x_co,
    name="planting_labor_requirement"
)

# 12. Total workers limit
model.addConstr(w_w + w_c + w_s + w_co <= Total_Number_of_Workers, name="total_workers_limit")

# ========== Solve ==========
model.optimize()

# ========== Results ==========
if model.status == GRB.OPTIMAL:
    print("Optimal solution found")
    print(f"Maximum profit: {model.objVal:.2f} yuan")
    print("\nPlanting areas (hectares):")
    print(f"  Wheat: {x_w.x:.2f}")
    print(f"  Corn: {x_c.x:.2f}")
    print(f"  Soybean: {x_s.x:.2f}")
    print(f"  Cotton: {x_co.x:.2f}")
    print("\nWorkers assigned:")
    print(f"  Wheat: {w_w.x:.0f}")
    print(f"  Corn: {w_c.x:.0f}")
    print(f"  Soybean: {w_s.x:.0f}")
    print(f"  Cotton: {w_co.x:.0f}")
    print(f"\nTotal workers used: {w_w.x + w_c.x + w_s.x + w_co.x:.0f}")
    print(f"\nResource usage:")
    print(f"  Irrigation water: {W.x:.2f} thousand m³")
    print(f"  Fertilizer: {F.x:.2f} tons")
    print(f"  Additional water labor: {lab_w.x:.2f} man-hours")
    print(f"  Additional fertilizer labor: {lab_f.x:.2f} man-hours")
    
    # Final answer output
    print(f"\nFinalAnswer=【{model.objVal:.2f}】")
else:
    print("No optimal solution found")
    print(f"FinalAnswer=【0】")