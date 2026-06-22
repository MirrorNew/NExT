import gurobipy as gp
from gurobipy import GRB

# Parameter List
number_of_areas = 12
weight_range_6_8 = [6, 8]
min_coverage_6_8 = 2
weight_threshold_9 = 9
min_coverage_9_up = 3
min_coverage_other = 1
max_fire_stations = 6
library_areas = [5, 6]
additional_coverage_for_library_areas = 1
num_alternative_sites = 8
# CoverableAreas and ConstructionCost are 1-indexed (using None at index 0)
CoverableAreas = [None, [1, 2, 5, 6, 7, 8, 11], [2, 3, 5, 6, 7], [1, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7], [6, 7, 10, 11], [4, 5, 8, 11, 12], [2, 9, 11, 12], [5, 10, 12]]
ConstructionCost = [None, 90, 70, 85, 65, 60, 80, 60, 50]
# ImportanceWeight is also 1-indexed
ImportanceWeight = [None, 4, 8, 5, 10, 7, 9, 6, 6, 3, 5, 8, 4]

# Create the model
model = gp.Model("Xunan_Fire_Station_Planning")

# Decision Variables
# x_i = 1 if a fire station is built at alternative site i, 0 otherwise
x = model.addVars(range(1, num_alternative_sites + 1), vtype=GRB.BINARY, name="x")

# Set up the objective function: Minimize total construction cost
model.setObjective(gp.quicksum(ConstructionCost[i] * x[i] for i in range(1, num_alternative_sites + 1)), GRB.MINIMIZE)

# Add all constraints
# 1. Redundant coverage requirements based on regional weights and special facilities (libraries)
for j in range(1, number_of_areas + 1):
    w_j = ImportanceWeight[j]
    
    # Identify the base requirement based on importance weight
    if weight_range_6_8[0] <= w_j <= weight_range_6_8[1]:
        base_req = min_coverage_6_8
    elif w_j >= weight_threshold_9:
        base_req = min_coverage_9_up
    else:
        base_req = min_coverage_other
        
    # Add extra coverage requirement for areas with libraries
    total_req = base_req
    if j in library_areas:
        total_req += additional_coverage_for_library_areas
        
    # Determine which sites can cover area j
    covering_sites = [i for i in range(1, num_alternative_sites + 1) if j in CoverableAreas[i]]
    
    # Add the constraint: Area j must be covered by at least 'total_req' sites
    model.addConstr(gp.quicksum(x[i] for i in covering_sites) >= total_req, name=f"Coverage_Req_Area_{j}")

# 2. Maximum number of stations allowed to be built
model.addConstr(gp.quicksum(x[i] for i in range(1, num_alternative_sites + 1)) <= max_fire_stations, name="Max_Stations_Constraint")

# Solve the model
model.optimize()

# Output the results
if model.status == GRB.OPTIMAL:
    objective_value = int(model.objVal)
    print(f"Optimal Construction Cost: {objective_value}")
    print("Fire stations built at sites:")
    for i in range(1, num_alternative_sites + 1):
        if x[i].X > 0.5:
            print(f"Site {i}")
    print(f"FinalAnswer=【{objective_value}】")
else:
    print("No optimal solution found.")