import gurobipy as gp

# ========== 2. Define parameters ==========
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

# Extract from Table_1_CostData
cost_data = [
    {'Product': 'Fertilizer', 'Profit_per_ton': 200, 'Raw_materials_per_ton': 0.5, 'Machine_time_per_ton': 0.8, 'Labor_per_ton': 0.6},
    {'Product': 'Paint', 'Profit_per_ton': 300, 'Raw_materials_per_ton': 0.7, 'Machine_time_per_ton': 1.0, 'Labor_per_ton': 0.8},
    {'Product': 'Chemicals', 'Profit_per_ton': 250, 'Raw_materials_per_ton': 0.6, 'Machine_time_per_ton': 0.9, 'Labor_per_ton': 0.7}
]

# Create dictionaries for quick access by product code
profit = {cost_data[i]['Product'][0]: cost_data[i]['Profit_per_ton'] for i in range(len(cost_data))}
raw_mat_per_ton = {cost_data[i]['Product'][0]: cost_data[i]['Raw_materials_per_ton'] for i in range(len(cost_data))}
machine_time_per_ton = {cost_data[i]['Product'][0]: cost_data[i]['Machine_time_per_ton'] for i in range(len(cost_data))}
labor_per_ton = {cost_data[i]['Product'][0]: cost_data[i]['Labor_per_ton'] for i in range(len(cost_data))}

# Compute total capacities from parameters
raw_material_capacity = suppliers * capacity_per_supplier  # 10 * 500 = 5000
machine_time_capacity = 2 * shift_hours * reactor_max_days  # 2 * 40 * 125 = 10000
labor_capacity = operators * max_hours_per_operator  # 80 * 100 = 8000

# ========== 3. Create model ==========
model = gp.Model("EACC_Production_Optimization")

# ========== 4. Decision variables ==========
x_F = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="x_F")
x_P = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="x_P")
x_C = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="x_C")

# ========== 5. Objective function ==========
model.setObjective(profit['F'] * x_F + profit['P'] * x_P + profit['C'] * x_C, sense=gp.GRB.MAXIMIZE)

# ========== 6. Constraints ==========
# 6.1 Raw-material supply
model.addConstr(
    raw_mat_per_ton['F'] * x_F + raw_mat_per_ton['P'] * x_P + raw_mat_per_ton['C'] * x_C <= raw_material_capacity,
    name="raw_material_limit"
)

# 6.2 Machine-time capacity
model.addConstr(
    machine_time_per_ton['F'] * x_F + machine_time_per_ton['P'] * x_P + machine_time_per_ton['C'] * x_C <= machine_time_capacity,
    name="machine_time_limit"
)

# 6.3 Per-product upper bound (general limit)
model.addConstr(x_F <= max_output_per_product, name="max_output_F")
model.addConstr(x_P <= max_output_per_product, name="max_output_P")
model.addConstr(x_C <= max_output_per_product, name="max_output_C")

# 6.4 Labor capacity
model.addConstr(
    labor_per_ton['F'] * x_F + labor_per_ton['P'] * x_P + labor_per_ton['C'] * x_C <= labor_capacity,
    name="labor_limit"
)

# 6.5 Demand limits
model.addConstr(x_F <= demand_max['F'], name="demand_limit_F")
model.addConstr(x_P <= demand_max['P'], name="demand_limit_P")
model.addConstr(x_C <= demand_max['C'], name="demand_limit_C")

# 6.6 Fertilizer share: x_F >= 0.25*(x_F + x_P + x_C)
model.addConstr(
    x_F >= fertilizer_min_ratio * (x_F + x_P + x_C),
    name="fertilizer_min_share"
)

# 6.7 Paint minimum
model.addConstr(x_P >= paint_min_output, name="paint_min")

# 6.8 Chemicals-to-fertilizer ratio: x_C <= 0.8 * x_F
model.addConstr(
    x_C <= chemical_max_to_fertilizer_ratio * x_F,
    name="chemical_to_fertilizer_ratio"
)

# 6.9 Average raw-material consumption: (0.5x_F+0.7x_P+0.6x_C) <= 0.65*(x_F+x_P+x_C)
model.addConstr(
    raw_mat_per_ton['F'] * x_F + raw_mat_per_ton['P'] * x_P + raw_mat_per_ton['C'] * x_C
    <= raw_material_consumption_limit * (x_F + x_P + x_C),
    name="avg_raw_material_limit"
)

# ========== 7. Solve and output results ==========
model.optimize()

if model.status == gp.GRB.OPTIMAL:
    print(f"Optimal objective value: ${model.ObjVal:.2f}")
    print(f"Production plan:")
    print(f"  Fertilizer (F): {x_F.X:.2f} tons")
    print(f"  Paint (P): {x_P.X:.2f} tons")
    print(f"  Chemicals (C): {x_C.X:.2f} tons")
    
    # Compute total profit
    total_profit = profit['F'] * x_F.X + profit['P'] * x_P.X + profit['C'] * x_C.X
    print(f"FinalAnswer=【{total_profit:.2f}】")
else:
    print(f"FinalAnswer=【No feasible solution found】")