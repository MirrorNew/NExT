import gurobipy as gp
from gurobipy import GRB

# 1. Initialize Model
model = gp.Model("FireStationOptimization")

# 2. Parameters List
number_of_areas = 12
weight_range_6_8 = [6, 8]
min_coverage_6_8 = 2
weight_threshold_9 = 9
min_coverage_9_up = 3
min_coverage_other = 1
max_fire_stations = 6
library_areas = [5, 6]
additional_coverage_for_library_areas = 1
coverage_time_limit = 5
num_alternative_sites = 8

# Note: Index 0 is None to align with 1-based indexing of sites and areas
CoverableAreas = [
    None, 
    [1, 2, 5, 6, 7, 8, 11], 
    [2, 3, 5, 6, 7], 
    [1, 4, 5, 6, 7, 8], 
    [3, 4, 5, 6, 7], 
    [6, 7, 10, 11], 
    [4, 5, 8, 11, 12], 
    [2, 9, 11, 12], 
    [5, 10, 12]
]
ConstructionCost = [None, 90, 70, 85, 65, 60, 80, 60, 50]
ImportanceWeight = [None, 4, 8, 5, 10, 7, 9, 6, 6, 3, 5, 8, 4]

# 3. Decision Variables
# x[i] = 1 if fire station is built at candidate site i, 0 otherwise
x = model.addVars(range(1, num_alternative_sites + 1), vtype=GRB.BINARY, name="x")

# 4. Objective Function
# Minimize total construction costs
model.setObjective(
    gp.quicksum(ConstructionCost[i] * x[i] for i in range(1, num_alternative_sites + 1)), 
    GRB.MINIMIZE
)

# 5. Constraints

# Pre-process coverage matrix: which sites cover area j?
# sites_covering[j] contains a list of site indices that cover area j
sites_covering = {j: [] for j in range(1, number_of_areas + 1)}
for i in range(1, num_alternative_sites + 1):
    if CoverableAreas[i] is not None:
        for area in CoverableAreas[i]:
            sites_covering[area].append(i)

# Coverage Constraints based on weights and library presence
for j in range(1, number_of_areas + 1):
    weight = ImportanceWeight[j]
    
    # Determine base requirement
    if weight >= weight_threshold_9:
        req = min_coverage_9_up
    elif weight_range_6_8[0] <= weight <= weight_range_6_8[1]:
        req = min_coverage_6_8
    else:
        req = min_coverage_other
        
    # Add library requirement
    if j in library_areas:
        req += additional_coverage_for_library_areas
        
    # Sum of variables for sites covering this area >= requirement
    model.addConstr(
        gp.quicksum(x[i] for i in sites_covering[j]) >= req,
        name=f"Coverage_Area_{j}_Req_{req}"
    )

# Maximum number of stations constraint
model.addConstr(
    gp.quicksum(x[i] for i in range(1, num_alternative_sites + 1)) <= max_fire_stations,
    name="Max_stations_limit"
)

# 6. Solve the model
model.optimize()

# 7. Print Results
if model.status == GRB.OPTIMAL:
    print("\nOptimal Solution Found:")
    built_sites = []
    for i in range(1, num_alternative_sites + 1):
        if x[i].x > 0.5:
            built_sites.append(i)
            print(f"Build Fire Station at Site {i} (Cost: {ConstructionCost[i]})")
    
    print(f"\nTotal Cost: {model.objVal}")
    print(f"Sites selected: {built_sites}")
    
    # Output the final answer in the required format
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found.")
    print(f"FinalAnswer=【No Solution】")