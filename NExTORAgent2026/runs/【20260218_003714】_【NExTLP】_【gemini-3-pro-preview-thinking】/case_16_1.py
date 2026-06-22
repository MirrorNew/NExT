import gurobipy as gp
from gurobipy import GRB

# Parameters List
number_of_candidate_sites = 8
number_of_user_areas = 10
total_construction_budget = 160
max_selected_sites = 5
average_construction_period_years = 1
basic_population_service_cost = 0.5
subway_equipment_cost = 1
subway_equipment_cost_reduction = 0.1
coverable_areas = [[], [1, 2, 5], [2, 3, 6], [1, 4, 7], [3, 5, 6, 7], [4, 6, 8, 9], [5, 7, 8, 10], [8, 9, 10], [9, 10]]
base_station_construction_cost = [0, 27, 25, 25, 30, 30, 30, 25, 20]
population = [0, 4, 8, 5, 10, 12, 7, 9, 3, 6, 11]
is_there_subway = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1]

# Create Model
model = gp.Model("BaseStationOptimization")

# Indices
sites = range(1, number_of_candidate_sites + 1) # 1..8
areas = range(1, number_of_user_areas + 1) # 1..10

# Pre-processing Coverage Matrix a_ij
# a[i, j] = 1 if site j covers area i
a = {}
for i in areas:
    for j in sites:
        a[i, j] = 0

for j in sites:
    covered_list = coverable_areas[j]
    for i in covered_list:
        a[i, j] = 1

# Pre-compute M_i for Big-M constraints (sum of a_ij for each i)
M = {}
for i in areas:
    M[i] = sum(a[i, j] for j in sites)

# Decision Variables
# x_j: 1 if site j is selected
x = model.addVars(sites, vtype=GRB.BINARY, name="x")
# y_i: 1 if user area i is covered
y = model.addVars(areas, vtype=GRB.BINARY, name="y")
# e_i: 1 if subway equipment is installed in area i
e = model.addVars(areas, vtype=GRB.BINARY, name="e")

# Objective Function
# Minimize total cost
# Z = Construction_Cost + Population_Service_Cost + Equipment_Cost
# Population_Service_Cost = sum( (0.5 * pop_i * y_i) - (0.1 * pop_i * e_i) )
# Equipment_Cost = sum(e_i)
obj_expr = gp.LinExpr()

# Construction Cost
for j in sites:
    obj_expr += base_station_construction_cost[j] * x[j]

# Service and Equipment Costs
for i in areas:
    term_service = basic_population_service_cost * population[i] * y[i]
    term_reduction = subway_equipment_cost_reduction * population[i] * e[i]
    term_eq_cost = subway_equipment_cost * e[i]
    obj_expr += term_service - term_reduction + term_eq_cost

model.setObjective(obj_expr, GRB.MINIMIZE)

# Constraints

# 1. Site-selection limit
model.addConstr(gp.quicksum(x[j] for j in sites) <= max_selected_sites, "SiteLimit")

# 2. Total budget
construction_cost_expr = gp.quicksum(base_station_construction_cost[j] * x[j] for j in sites)
equipment_cost_expr = gp.quicksum(subway_equipment_cost * e[i] for i in areas)
model.addConstr(construction_cost_expr + equipment_cost_expr <= total_construction_budget, "Budget")

# 3. Coverage logic (Lower bound)
# sum(a_ij * x_j) >= y_i
for i in areas:
    model.addConstr(gp.quicksum(a[i, j] * x[j] for j in sites) >= y[i], f"CoverageLower_{i}")

# 4. Coverage logic (Upper bound)
# sum(a_ij * x_j) <= M_i * y_i
# Forces y_i = 1 if area is covered by any station
for i in areas:
    model.addConstr(gp.quicksum(a[i, j] * x[j] for j in sites) <= M[i] * y[i], f"CoverageUpper_{i}")

# 5. Global coverage requirement
# Each area must be covered by at least one base station
for i in areas:
    model.addConstr(gp.quicksum(a[i, j] * x[j] for j in sites) >= 1, f"GlobalCoverage_{i}")

# 6. Interference in areas 8, 9, 10
# Can be covered by at most one base station
interference_areas = [8, 9, 10]
for i in interference_areas:
    model.addConstr(gp.quicksum(a[i, j] * x[j] for j in sites) <= 1, f"Interference_{i}")

# 7. Equipment only where subway exists
for i in areas:
    model.addConstr(e[i] <= is_there_subway[i], f"SubwayEqLimit_{i}")

# Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print("\nOptimal Solution:")
    for j in sites:
        if x[j].x > 0.5:
            print(f"  Build Base Station {j}")
    
    print("\nCoverage Status:")
    for i in areas:
        if y[i].x > 0.5:
            # Check actual coverage count
            count = sum(a[i,j] * x[j].x for j in sites)
            print(f"  Area {i} Covered (by {count:.0f} stations)")
            
    print("\nSubway Equipment:")
    for i in areas:
        if e[i].x > 0.5:
            print(f"  Install Equipment in Area {i}")
            
    print(f"\nTotal Cost: {model.objVal}")
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("No feasible solution found.")