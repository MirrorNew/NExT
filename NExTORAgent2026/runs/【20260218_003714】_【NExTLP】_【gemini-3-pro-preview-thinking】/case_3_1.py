import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
# Parameters from the provided Parameters List
planning_year = 2025
planning_periods = 2
products_list = ['F', 'P', 'C']  # Renamed to avoid conflict with model logic
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

# Derived constants based on problem description
shifts_count = 2  # "double shifts"

# Organize cost data into dictionaries for easy access
# Mapping full names to abbreviations: Fertilizer->F, Paint->P, Chemicals->C
product_map = {'Fertilizer': 'F', 'Paint': 'P', 'Chemicals': 'C'}
profit_per_ton = {}
raw_mat_per_ton = {}
machine_time_per_ton = {}
labor_per_ton = {}

for entry in Table_1_CostData:
    p_code = product_map[entry['Product']]
    profit_per_ton[p_code] = entry['Profit_per_ton']
    raw_mat_per_ton[p_code] = entry['Raw_materials_per_ton']
    machine_time_per_ton[p_code] = entry['Machine_time_per_ton']
    labor_per_ton[p_code] = entry['Labor_per_ton']

# Calculate Total Capacities
total_raw_material_supply = suppliers * capacity_per_supplier
total_machine_capacity = reactor_max_days * shifts_count * shift_hours
total_labor_capacity = operators * max_hours_per_operator

# 2. Initialize Model
model = gp.Model("EAC_Production_Plan")

# 3. Create Decision Variables
# Continuous variables for production quantities of F, P, C
x = model.addVars(products_list, vtype=GRB.CONTINUOUS, lb=0, name="x")

# 4. Set up the Objective Function
# Maximize Total Profit
model.setObjective(
    gp.quicksum(profit_per_ton[p] * x[p] for p in products_list),
    GRB.MAXIMIZE
)

# 5. Add Constraints

# (1) Raw Material Supply Constraint
model.addConstr(
    gp.quicksum(raw_mat_per_ton[p] * x[p] for p in products_list) <= total_raw_material_supply,
    "RawMaterialSupply"
)

# (2) Machine Time Capacity Constraint
model.addConstr(
    gp.quicksum(machine_time_per_ton[p] * x[p] for p in products_list) <= total_machine_capacity,
    "MachineTimeCapacity"
)

# (3) Labor Capacity Constraint
model.addConstr(
    gp.quicksum(labor_per_ton[p] * x[p] for p in products_list) <= total_labor_capacity,
    "LaborCapacity"
)

# (4) Per-product Upper Bound (Production Capacity)
for p in products_list:
    model.addConstr(x[p] <= max_output_per_product, f"MaxOutput_{p}")

# (5) Demand Limits
# The demand max is per quarter, but the plan is for 'planning_periods' (2) quarters.
for p in products_list:
    model.addConstr(x[p] <= demand_max[p] * planning_periods, f"DemandLimit_{p}")

# (6) Fertilizer Share Constraint (at least 25% of total output)
# x_F >= 0.25 * (x_F + x_P + x_C)
model.addConstr(x['F'] >= fertilizer_min_ratio * x.sum(), "FertilizerShare")

# (7) Paint Minimum Production Constraint
model.addConstr(x['P'] >= paint_min_output, "PaintMinimum")

# (8) Chemicals-to-Fertilizer Ratio Constraint
# x_C <= 0.8 * x_F
model.addConstr(x['C'] <= chemical_max_to_fertilizer_ratio * x['F'], "ChemToFertRatio")

# (9) Average Raw Material Consumption Constraint
# (Total Raw Materials) / (Total Output) <= 0.65
# Linearized: Total Raw Materials <= 0.65 * Total Output
model.addConstr(
    gp.quicksum(raw_mat_per_ton[p] * x[p] for p in products_list) <= 
    raw_material_consumption_limit * x.sum(),
    "AvgRawMatConsumption"
)

# 6. Solve the model and print results
model.optimize()

if model.status == GRB.OPTIMIZED:
    print("\nOptimal Solution Found:")
    for p in products_list:
        print(f"Product {p}: {x[p].X} tons")
    
    print(f"\nTotal Profit: ${model.ObjVal:,.2f}")
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("No optimal solution found.")