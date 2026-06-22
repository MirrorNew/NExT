import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters (from Parameters List)
# =========================
planning_year = 2025
planning_periods = 2
products = ['F', 'P', 'C']
suppliers = 10
capacity_per_supplier = 500
reactor_max_days = 125
shift_hours = 40
max_output_per_product = 3000
operators = 80
max_hours_per_operator = 100
demand_max = {'F': 2000, 'P': 1500, 'C': 1800}
fertilizer_min_ratio = 0.25
paint_min_output = 1000
chemical_max_to_fertilizer_ratio = 0.8
raw_material_consumption_limit = 0.65
donation_ratio = 0.01
Table_1_CostData = [
    {'Product': 'Fertilizer', 'Profit_per_ton': 200, 'Raw_materials_per_ton': 0.5,
     'Machine_time_per_ton': 0.8, 'Labor_per_ton': 0.6},
    {'Product': 'Paint', 'Profit_per_ton': 300, 'Raw_materials_per_ton': 0.7,
     'Machine_time_per_ton': 1.0, 'Labor_per_ton': 0.8},
    {'Product': 'Chemicals', 'Profit_per_ton': 250, 'Raw_materials_per_ton': 0.6,
     'Machine_time_per_ton': 0.9, 'Labor_per_ton': 0.7}
]

# Map table data to product codes F, P, C
profit_per_ton = {'F': 200, 'P': 300, 'C': 250}
raw_materials_per_ton = {'F': 0.5, 'P': 0.7, 'C': 0.6}
machine_time_per_ton = {'F': 0.8, 'P': 1.0, 'C': 0.9}
labor_per_ton = {'F': 0.6, 'P': 0.8, 'C': 0.7}

# Derived capacities from parameters
total_raw_material_capacity = suppliers * capacity_per_supplier          # 10 * 500 = 5000
total_machine_hours_capacity = reactor_max_days * 2 * shift_hours        # 125 days * 2 shifts * 40 = 10000
total_labor_hours_capacity = operators * max_hours_per_operator          # 80 * 100 = 8000

# =========================
# 2. Create model
# =========================
model = gp.Model("EAC_Production_Planning_2025")

# =========================
# 3. Decision variables
# =========================
# x_F, x_P, x_C: production volumes (tons)
x = model.addVars(products, name="x", lb=0.0, vtype=GRB.CONTINUOUS)

# =========================
# 4. Objective function: Maximize total profit
#     Z = 200 x_F + 300 x_P + 250 x_C
# =========================
model.setObjective(
    profit_per_ton['F'] * x['F'] +
    profit_per_ton['P'] * x['P'] +
    profit_per_ton['C'] * x['C'],
    GRB.MAXIMIZE
)

# =========================
# 5. Constraints
# =========================

# 5.1 Raw-material supply
# 0.5 x_F + 0.7 x_P + 0.6 x_C ≤ 5000
model.addConstr(
    raw_materials_per_ton['F'] * x['F'] +
    raw_materials_per_ton['P'] * x['P'] +
    raw_materials_per_ton['C'] * x['C'] <= total_raw_material_capacity,
    name="RawMaterialCapacity"
)

# 5.2 Machine-time capacity
# 0.8 x_F + 1.0 x_P + 0.9 x_C ≤ 10000
model.addConstr(
    machine_time_per_ton['F'] * x['F'] +
    machine_time_per_ton['P'] * x['P'] +
    machine_time_per_ton['C'] * x['C'] <= total_machine_hours_capacity,
    name="MachineTimeCapacity"
)

# 5.3 Per-product upper bounds (3000 each)
# x_F ≤ 3000; x_P ≤ 3000; x_C ≤ 3000
for p in products:
    model.addConstr(x[p] <= max_output_per_product, name=f"PerProductUpper_{p}")

# 5.4 Labor capacity
# 0.6 x_F + 0.8 x_P + 0.7 x_C ≤ 8000
model.addConstr(
    labor_per_ton['F'] * x['F'] +
    labor_per_ton['P'] * x['P'] +
    labor_per_ton['C'] * x['C'] <= total_labor_hours_capacity,
    name="LaborCapacity"
)

# 5.5 Demand limits
# x_F ≤ 2000; x_P ≤ 1500; x_C ≤ 1800
model.addConstr(x['F'] <= demand_max['F'], name="Demand_F")
model.addConstr(x['P'] <= demand_max['P'], name="Demand_P")
model.addConstr(x['C'] <= demand_max['C'], name="Demand_C")

# 5.6 Fertilizer share
# x_F ≥ 0.25 (x_F + x_P + x_C)
# -> x_F ≥ fertilizer_min_ratio * (x_F + x_P + x_C)
model.addConstr(
    x['F'] >= fertilizer_min_ratio * (x['F'] + x['P'] + x['C']),
    name="FertilizerShare"
)

# 5.7 Minimum paint production
# x_P ≥ 1000
model.addConstr(
    x['P'] >= paint_min_output,
    name="PaintMinimum"
)

# 5.8 Chemicals-to-fertilizer ratio
# x_C ≤ 0.8 x_F
model.addConstr(
    x['C'] <= chemical_max_to_fertilizer_ratio * x['F'],
    name="ChemicalsToFertilizerRatio"
)

# 5.9 Average raw-material consumption
# 0.5 x_F + 0.7 x_P + 0.6 x_C ≤ 0.65 (x_F + x_P + x_C)
model.addConstr(
    raw_materials_per_ton['F'] * x['F'] +
    raw_materials_per_ton['P'] * x['P'] +
    raw_materials_per_ton['C'] * x['C']
    <= raw_material_consumption_limit * (x['F'] + x['P'] + x['C']),
    name="AvgRawMaterialConsumption"
)

# =========================
# 6. Optimize
# =========================
model.optimize()

# =========================
# 7. Print results
# =========================
if model.Status == GRB.OPTIMAL:
    x_F_opt = x['F'].X
    x_P_opt = x['P'].X
    x_C_opt = x['C'].X
    optimal_profit = model.ObjVal

    print(f"Optimal solution found for planning year {planning_year}:")
    print(f"  Fertilizer production (x_F): {x_F_opt:.4f} tons")
    print(f"  Paint production (x_P):      {x_P_opt:.4f} tons")
    print(f"  Chemicals production (x_C):  {x_C_opt:.4f} tons")
    print(f"  Maximum total profit:        {optimal_profit:.2f} USD")

    # FinalAnswer should only be the maximum total profit
    print(f"FinalAnswer=【{optimal_profit:.2f}】")
else:
    print("No optimal solution found.")
    print("FinalAnswer=【NaN】")