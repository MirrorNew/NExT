import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and any other necessary packages.
# Already imported gp and GRB.

# 2. Define all parameter matrices and data inputs.
# Parameters given in the question and parameters list:
suppliers = 10
capacity_per_supplier = 500
total_raw_material_supply = suppliers * capacity_per_supplier  # 5000 tons

# Machine Capacity calculation: 125 days * 2 shifts/day * 40 hours/shift = 10000 machine-hours
total_machine_hours = 10000 

operators = 80
max_hours_per_operator = 100
total_labor_hours = operators * max_hours_per_operator  # 8000 hours

# Product-specific limits (Demand and Capacity)
# Demand limits are tighter than the 3000-ton capacity cap
demand_max = {'F': 2000, 'P': 1500, 'C': 1800}
max_output_per_product = 3000

# Other ratios/requirements
fertilizer_min_ratio = 0.25
paint_min_output = 1000
chemical_max_to_fertilizer_ratio = 0.8
raw_material_consumption_limit = 0.65

# Profit per ton (already adjusted for 1% donation)
# From Table 1:
# Fertilizer: 200
# Paint: 300
# Chemicals: 250
# Raw material/ton: F: 0.5, P: 0.7, C: 0.6
# Machine time/ton: F: 0.8, P: 1.0, C: 0.9
# Labor time/ton: F: 0.6, P: 0.8, C: 0.7

# 3. Create a model
model = gp.Model("EAC_Production_Optimization")

# 4. Create decision variables.
x_F = model.addVar(lb=0, name="x_F")
x_P = model.addVar(lb=0, name="x_P")
x_C = model.addVar(lb=0, name="x_C")

# 5. Set up the objective function.
model.setObjective(200 * x_F + 300 * x_P + 250 * x_C, GRB.MAXIMIZE)

# 6. Add all constraints.
# Raw-material supply
model.addConstr(0.5 * x_F + 0.7 * x_P + 0.6 * x_C <= total_raw_material_supply, "Raw_Material_Supply")

# Machine-time capacity
model.addConstr(0.8 * x_F + 1.0 * x_P + 0.9 * x_C <= total_machine_hours, "Machine_Time_Capacity")

# Labor capacity
model.addConstr(0.6 * x_F + 0.8 * x_P + 0.7 * x_C <= total_labor_hours, "Labor_Capacity")

# Per-product upper bound (Capacity & Demand)
# The demand limit is the more restrictive constraint
model.addConstr(x_F <= demand_max['F'], "Demand_F")
model.addConstr(x_P <= demand_max['P'], "Demand_P")
model.addConstr(x_C <= demand_max['C'], "Demand_C")
model.addConstr(x_F <= max_output_per_product, "Cap_F")
model.addConstr(x_P <= max_output_per_product, "Cap_P")
model.addConstr(x_C <= max_output_per_product, "Cap_C")

# Fertilizer share requirement: x_F >= 0.25 * (x_F + x_P + x_C)
model.addConstr(x_F >= fertilizer_min_ratio * (x_F + x_P + x_C), "Fertilizer_Share")

# Paint minimum output
model.addConstr(x_P >= paint_min_output, "Paint_Minimum")

# Chemicals-to-fertilizer ratio: x_C <= 0.8 * x_F
model.addConstr(x_C <= chemical_max_to_fertilizer_ratio * x_F, "Chemical_to_Fertilizer_Ratio")

# Avg. raw-material consumption: (0.5 x_F + 0.7 x_P + 0.6 x_C) / (x_F + x_P + x_C) <= 0.65
model.addConstr(0.5 * x_F + 0.7 * x_P + 0.6 * x_C <= raw_material_consumption_limit * (x_F + x_P + x_C), "Avg_Raw_Material")

# 7. Solve the model and print results.
model.optimize()

if model.status == GRB.OPTIMAL:
    objective_value = model.objVal
    print(f"FinalAnswer=【{objective_value}】")