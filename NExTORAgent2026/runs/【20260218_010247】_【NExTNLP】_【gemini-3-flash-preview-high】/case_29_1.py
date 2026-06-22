import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and any other necessary packages.
# (Done above)

# 2. Define all parameter matrices and data inputs.
threshold_speed_extra = 50
number_of_sections = 3
base_fuel_consumption_coefficient = 0.005
extra_fuel_consumption_coefficient = 0.0001
total_sailing_time_limit = 10
min_speed_section_2 = 55

# Table 1: Section Characteristics
# Distance, Fuel Price, Maximum Speed Limit
sections_data = [
    {'dist': 200, 'price': 3000, 'max_v': 70},
    {'dist': 100, 'price': 3500, 'max_v': 60},
    {'dist': 150, 'price': 4000, 'max_v': 80}
]

# Create the model
model = gp.Model("ShipSpeedOptimization")

# 3. Create decision variables.
v1 = model.addVar(lb=1, ub=sections_data[0]['max_v'], name="v1")
v2 = model.addVar(lb=min_speed_section_2, ub=sections_data[1]['max_v'], name="v2")
v3 = model.addVar(lb=1, ub=sections_data[2]['max_v'], name="v3")

d1 = model.addVar(lb=0, ub=sections_data[0]['max_v'] - threshold_speed_extra, name="d1")
d2 = model.addVar(lb=0, ub=sections_data[1]['max_v'] - threshold_speed_extra, name="d2")
d3 = model.addVar(lb=0, ub=sections_data[2]['max_v'] - threshold_speed_extra, name="d3")

# Indicator variables for sections where speed can be above or below 50 km/h
y1 = model.addVar(vtype=GRB.BINARY, name="y1")
y3 = model.addVar(vtype=GRB.BINARY, name="y3")

# 4. Create any auxiliary substitution variables (lb=-GRB.INFINITY, ub=GRB.INFINITY).
v1_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="v1_sq")
v2_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="v2_sq")
v3_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="v3_sq")

d1_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="d1_sq")
d2_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="d2_sq")
d3_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="d3_sq")

inv_v1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="inv_v1")
inv_v2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="inv_v2")
inv_v3 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="inv_v3")

# 5. Set up the objective function.
# Z = sum_{i=1 to 3} [P_i * D_i * (0.005 * v_i^2 + 0.0001 * d_i^2)]
cost1 = sections_data[0]['price'] * sections_data[0]['dist'] * (base_fuel_consumption_coefficient * v1_sq + extra_fuel_consumption_coefficient * d1_sq)
cost2 = sections_data[1]['price'] * sections_data[1]['dist'] * (base_fuel_consumption_coefficient * v2_sq + extra_fuel_consumption_coefficient * d2_sq)
cost3 = sections_data[2]['price'] * sections_data[2]['dist'] * (base_fuel_consumption_coefficient * v3_sq + extra_fuel_consumption_coefficient * d3_sq)
model.setObjective(cost1 + cost2 + cost3, GRB.MINIMIZE)

# 6. Add all constraints.
# Quadratic auxiliary constraints
model.addGenConstrPow(v1, v1_sq, 2)
model.addGenConstrPow(v2, v2_sq, 2)
model.addGenConstrPow(v3, v3_sq, 2)
model.addGenConstrPow(d1, d1_sq, 2)
model.addGenConstrPow(d2, d2_sq, 2)
model.addGenConstrPow(d3, d3_sq, 2)

# Reciprocal auxiliary constraints to handle time constraint denominators
model.addConstr(v1 * inv_v1 == 1)
model.addConstr(v2 * inv_v2 == 1)
model.addConstr(v3 * inv_v3 == 1)

# Total time constraint: D1/v1 + D2/v2 + D3/v3 <= 10
model.addConstr(sections_data[0]['dist'] * inv_v1 + sections_data[1]['dist'] * inv_v2 + sections_data[2]['dist'] * inv_v3 <= total_sailing_time_limit)

# Definition of additional speed d2 for Section 2 (where v2 >= 55)
model.addConstr(d2 == v2 - threshold_speed_extra)

# Definition of additional speed d1 and d3 using Indicator constraints
# Section 1 logic
model.addGenConstrIndicator(y1, 1, v1 >= 50)
model.addGenConstrIndicator(y1, 0, v1 <= 50)
model.addGenConstrIndicator(y1, 1, v1 - d1 == 50)
model.addGenConstrIndicator(y1, 0, d1 == 0)

# Section 3 logic
model.addGenConstrIndicator(y3, 1, v3 >= 50)
model.addGenConstrIndicator(y3, 0, v3 <= 50)
model.addGenConstrIndicator(y3, 1, v3 - d3 == 50)
model.addGenConstrIndicator(y3, 0, d3 == 0)

# 7. Solve the model and print results.
model.Params.NonConvex = 2
model.optimize()

if model.Status == GRB.OPTIMAL:
    print(f"FinalAnswer=【{model.ObjVal}】")