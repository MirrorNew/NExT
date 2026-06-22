import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
# Parameters from the provided Parameters List
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

# Table 1 Data
Table_1_CostData = [
    {'Product': 'Fertilizer', 'Profit_per_ton': 200, 'Raw_materials_per_ton': 0.5, 'Machine_time_per_ton': 0.8, 'Labor_per_ton': 0.6},
    {'Product': 'Paint', 'Profit_per_ton': 300, 'Raw_materials_per_ton': 0.7, 'Machine_time_per_ton': 1.0, 'Labor_per_ton': 0.8},
    {'Product': 'Chemicals', 'Profit_per_ton': 250, 'Raw_materials_per_ton': 0.6, 'Machine_time_per_ton': 0.9, 'Labor_per_ton': 0.7}
]

# Map product codes to data for easier access
product_data = {
    'F': [d for d in Table_1_CostData if d['Product'] == 'Fertilizer'][0],
    'P': [d for d in Table_1_CostData if d['Product'] == 'Paint'][0],
    'C': [d for d in Table_1_CostData if d['Product'] == 'Chemicals'][0]
}

# Derived Capacities
# Raw Material: 10 suppliers * 500 tons
total_raw_material_supply = suppliers * capacity_per_supplier

# Machine Time: 125 days * 2 shifts * 40 hours per shift
# Note: "double shifts" implies a multiplier of 2
total_machine_capacity = reactor_max_days * 2 * shift_hours

# Labor: 80 operators * 100 hours
total_labor_capacity = operators * max_hours_per_operator

# 2. Initialize Model
model = gp.Model("EAC_Production_Optimization")

# 3. Create Decision Variables
x_F = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x_F")
x_P = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x_P")
x_C = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x_C")

# 4. Set up the Objective Function
# Maximize Total Profit
# Note: Donation is already deducted from the profit coefficient in the table
model.setObjective(
    product_data['F']['Profit_per_ton'] * x_F + 
    product_data['P']['Profit_per_ton'] * x_P + 
    product_data['C']['Profit_per_ton'] * x_C, 
    GRB.MAXIMIZE
)

# 5. Add Constraints

# (1) Raw Material Supply Constraint
model.addConstr(
    product_data['F']['Raw_materials_per_ton'] * x_F + 
    product_data['P']['Raw_materials_per_ton'] * x_P + 
    product_data['C']['Raw_materials_per_ton'] * x_C <= total_raw_material_supply, 
    "RawMaterialSupply"
)

# (2) Machine Time Capacity Constraint
model.addConstr(
    product_data['F']['Machine_time_per_ton'] * x_F + 
    product_data['P']['Machine_time_per_ton'] * x_P + 
    product_data['C']['Machine_time_per_ton'] * x_C <= total_machine_capacity, 
    "MachineTimeCapacity"
)

# (3) Labor Capacity Constraint
model.addConstr(
    product_data['F']['Labor_per_ton'] * x_F + 
    product_data['P']['Labor_per_ton'] * x_P + 
    product_data['C']['Labor_per_ton'] * x_C <= total_labor_capacity, 
    "LaborCapacity"
)

# (4) Per-product Upper Bound
model.addConstr(x_F <= max_output_per_product, "MaxOutput_F")
model.addConstr(x_P <= max_output_per_product, "MaxOutput_P")
model.addConstr(x_C <= max_output_per_product, "MaxOutput_C")

# (5) Demand Limits
# The plan covers "the next two quarters", so demand limits are scaled by planning_periods (2).
model.addConstr(x_F <= demand_max['F'] * planning_periods, "Demand_F")
model.addConstr(x_P <= demand_max['P'] * planning_periods, "Demand_P")
model.addConstr(x_C <= demand_max['C'] * planning_periods, "Demand_C")

# (6) Fertilizer Share Constraint
# x_F >= 0.25 * (x_F + x_P + x_C)
model.addConstr(x_F >= fertilizer_min_ratio * (x_F + x_P + x_C), "FertilizerShare")

# (7) Paint Minimum Production
model.addConstr(x_P >= paint_min_output, "PaintMin")

# (8) Chemicals-to-Fertilizer Ratio
# x_C <= 0.8 * x_F
model.addConstr(x_C <= chemical_max_to_fertilizer_ratio * x_F, "ChemicalToFertilizerRatio")

# (9) Average Raw Material Consumption
# (Total Raw Materials) / (Total Output) <= 0.65
lhs_raw = (product_data['F']['Raw_materials_per_ton'] * x_F + 
           product_data['P']['Raw_materials_per_ton'] * x_P + 
           product_data['C']['Raw_materials_per_ton'] * x_C)
rhs_raw = raw_material_consumption_limit * (x_F + x_P + x_C)
model.addConstr(lhs_raw <= rhs_raw, "AvgRawMaterialConsumption")

# 6. Solve the model and print results
model.optimize()

if model.status == GRB.OPTIMIZED:
    print(f"Optimal Solution Found:")
    print(f"Fertilizer (x_F): {x_F.X}")
    print(f"Paint (x_P): {x_P.X}")
    print(f"Chemicals (x_C): {x_C.X}")
    print(f"Total Profit: {model.ObjVal}")
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("No optimal solution found.")