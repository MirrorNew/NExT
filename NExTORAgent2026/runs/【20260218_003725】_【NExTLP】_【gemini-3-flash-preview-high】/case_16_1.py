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
site_costs = [0, 27, 25, 25, 30, 30, 30, 25, 20]

# Population by area (1-indexed)
population = [0, 4, 8, 5, 10, 12, 7, 9, 3, 6, 11]

# Subway availability by area (1-indexed)
is_subway = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1]

# Base station coverage relationships (Site: Areas it covers)
site_to_areas = {
    1: [1, 2, 5],
    2: [2, 3, 6],
    3: [1, 4, 7],
    4: [3, 5, 6, 7],
    5: [4, 6, 8, 9],
    6: [5, 7, 8, 10],
    7: [8, 9, 10],
    8: [9, 10]
}

# Mapping areas back to the sites that can cover them
area_to_sites = {i: [] for i in range(1, num_user_areas + 1)}
for s, areas in site_to_areas.items():
    for a in areas:
        area_to_sites[a].append(s)

# Create the Gurobi model
model = gp.Model("Dongjiang_Base_Station_Layout")

# Decision Variables
# x[j] = 1 if site j is selected for construction, 0 otherwise
x = model.addVars(range(1, num_candidate_sites + 1), vtype=GRB.BINARY, name="x")

# y[i] = 1 if area i is covered, 0 otherwise
y = model.addVars(range(1, num_user_areas + 1), vtype=GRB.BINARY, name="y")

# e[i] = 1 if subway equipment is installed in area i, 0 otherwise
e = model.addVars(range(1, num_user_areas + 1), vtype=GRB.BINARY, name="e")

# Objective Function: Minimize Total Cost
# Total Cost = Site Construction + Population Service + Subway Equipment
# Service cost for area i is basic cost (0.5 * pop_i * y_i) reduced by equipment benefit (0.1 * pop_i * e_i)
construction_cost = gp.quicksum(site_costs[j] * x[j] for j in range(1, num_candidate_sites + 1))
population_service_cost = gp.quicksum(basic_population_service_cost * population[i] * y[i] - 
                                       subway_equipment_cost_reduction * population[i] * e[i] 
                                       for i in range(1, num_user_areas + 1))
subway_installation_cost = gp.quicksum(subway_equipment_cost * e[i] for i in range(1, num_user_areas + 1))

model.setObjective(construction_cost + population_service_cost + subway_installation_cost, GRB.MINIMIZE)

# Constraints

# 1. Site-selection limit: A maximum of 5 sites can be selected
model.addConstr(gp.quicksum(x[j] for j in range(1, num_candidate_sites + 1)) <= max_sites)

# 2. Total construction budget: Site construction + subway equipment <= 160
model.addConstr(construction_cost + subway_installation_cost <= total_budget)

# 3. Global coverage requirement: All areas must be covered
for i in range(1, num_user_areas + 1):
    model.addConstr(y[i] == 1)

# 4. Coverage logic: y[i] corresponds to whether at least one station covers area i
# Using Indicator Constraints as specified:
for i in range(1, num_user_areas + 1):
    # If y[i] = 1, then at least one site covering area i must be selected
    model.addGenConstrIndicator(y[i], 1, gp.quicksum(x[j] for j in area_to_sites[i]), GRB.GREATER_EQUAL, 1)
    # If y[i] = 0, then no site covering area i can be selected
    model.addGenConstrIndicator(y[i], 0, gp.quicksum(x[j] for j in area_to_sites[i]), GRB.EQUAL, 0)

# 5. Interference risk in areas 8, 9, 10: At most one base station can cover each of these areas
for i in [8, 9, 10]:
    model.addConstr(gp.quicksum(x[j] for j in area_to_sites[i]) <= 1)

# 6. Subway equipment restriction: Can only install where subways exist and areas are covered
for i in range(1, num_user_areas + 1):
    model.addConstr(e[i] <= is_subway[i])
    model.addConstr(e[i] <= y[i])

# Solve the model
model.optimize()

# Print results and output formatted FinalAnswer
if model.status == GRB.OPTIMAL:
    final_cost = model.objVal
    print(f"FinalAnswer=【{final_cost}】")
else:
    print("No optimal solution found.")