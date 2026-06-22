import gurobipy as gp
from gurobipy import GRB

# Parameters from the list
num_candidate_sites = 8
num_user_areas = 10
total_construction_budget = 160
max_selected_sites = 5
basic_population_service_cost = 0.5
subway_equipment_cost = 1
subway_equipment_cost_reduction = 0.1

# Site construction costs (1-indexed based on the parameter list)
# Index 0 is a placeholder to align with site numbering 1-8
site_costs = [0, 27, 25, 25, 30, 30, 30, 25, 20]

# Population by area (1-indexed)
# Index 0 is a placeholder to align with area numbering 1-10
population = [0, 4, 8, 5, 10, 12, 7, 9, 3, 6, 11]

# Subway availability by area (1-indexed)
is_there_subway = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1]

# Base station coverage relationships (Site: Areas it covers)
coverable_areas = [[], [1, 2, 5], [2, 3, 6], [1, 4, 7], [3, 5, 6, 7], [4, 6, 8, 9], [5, 7, 8, 10], [8, 9, 10], [9, 10]]

# Pre-calculate mapping: which sites cover each area
area_to_sites = {i: [] for i in range(1, num_user_areas + 1)}
for site_id in range(1, num_candidate_sites + 1):
    for area_id in coverable_areas[site_id]:
        area_to_sites[area_id].append(site_id)

# Create the Gurobi model
model = gp.Model("Dongjiang_Base_Station_Optimization")

# Decision Variables
# x[j] = 1 if site j is selected for construction, 0 otherwise
x = model.addVars(range(1, num_candidate_sites + 1), vtype=GRB.BINARY, name="x")

# y[i] = 1 if user area i is covered, 0 otherwise
y = model.addVars(range(1, num_user_areas + 1), vtype=GRB.BINARY, name="y")

# e[i] = 1 if subway equipment is installed in area i, 0 otherwise
e = model.addVars(range(1, num_user_areas + 1), vtype=GRB.BINARY, name="e")

# Objective Function: Minimize total cost
# Total Cost = Construction Costs + Population Service Costs + Subway Equipment Costs
construction_cost = gp.quicksum(site_costs[j] * x[j] for j in range(1, num_candidate_sites + 1))
subway_installation_cost = gp.quicksum(subway_equipment_cost * e[i] for i in range(1, num_user_areas + 1))
population_service_cost = gp.quicksum(basic_population_service_cost * population[i] * y[i] - 
                                       subway_equipment_cost_reduction * population[i] * e[i] 
                                       for i in range(1, num_user_areas + 1))

model.setObjective(construction_cost + population_service_cost + subway_installation_cost, GRB.MINIMIZE)

# Constraints

# 1. Site-selection limit: Max 5 sites
model.addConstr(gp.quicksum(x[j] for j in range(1, num_candidate_sites + 1)) <= max_selected_sites)

# 2. Total construction budget: Site construction + subway equipment <= 160
model.addConstr(construction_cost + subway_installation_cost <= total_construction_budget)

# 3. Global coverage requirement: All 10 areas must be covered
for i in range(1, num_user_areas + 1):
    model.addConstr(y[i] == 1)

# 4. Coverage logic: y[i] corresponds to whether at least one station covers area i
# Using Indicator Constraints as specified
for i in range(1, num_user_areas + 1):
    # If y[i] = 1, then at least one site covering area i must be selected
    model.addGenConstrIndicator(y[i], 1, gp.quicksum(x[j] for j in area_to_sites[i]), GRB.GREATER_EQUAL, 1)
    # If y[i] = 0, then no site covering area i can be selected
    model.addGenConstrIndicator(y[i], 0, gp.quicksum(x[j] for j in area_to_sites[i]), GRB.EQUAL, 0)

# 5. Interference risk in areas 8, 9, 10: At most one base station from the union set covers this cluster
# Union of sites covering area 8, 9, or 10 is {5, 6, 7, 8}
interference_union_sites = {5, 6, 7, 8}
model.addConstr(gp.quicksum(x[j] for j in interference_union_sites) <= 1)

# 6. Subway equipment restriction: Can only install where subways exist and areas are covered
for i in range(1, num_user_areas + 1):
    # Only install if area has subway
    model.addConstr(e[i] <= is_there_subway[i])
    # Only install if area is covered
    model.addConstr(e[i] <= y[i])

# Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    final_cost = model.objVal
    print(f"FinalAnswer=【{final_cost}】")
else:
    print("Optimization was not successful.")