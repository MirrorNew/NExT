import gurobipy as gp

# Parameters from the provided list
number_of_areas = 12
weight_range_6_8 = (6, 8)
min_coverage_6_8 = 2
weight_threshold_9 = 9
min_coverage_9_up = 3
min_coverage_other = 1
max_fire_stations = 6
library_areas = [5, 6]
additional_coverage_for_library_areas = 1
coverage_time_limit = 5
num_alternative_sites = 8

# CoverableAreas: index 0 is None, indices 1-8 correspond to sites 1-8
CoverableAreas = [None, [1, 2, 5, 6, 7, 8, 11], [2, 3, 5, 6, 7], [1, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7], [6, 7, 10, 11], [4, 5, 8, 11, 12], [2, 9, 11, 12], [5, 10, 12]]

# ConstructionCost: index 0 is None, indices 1-8 correspond to sites 1-8
ConstructionCost = [None, 90, 70, 85, 65, 60, 80, 60, 50]

# ImportanceWeight: index 0 is None, indices 1-12 correspond to areas 1-12
ImportanceWeight = [None, 4, 8, 5, 10, 7, 9, 6, 6, 3, 5, 8, 4]

# Create coverage matrix a[i][j] = 1 if site i covers area j
# Indices: i from 1 to 8, j from 1 to 12
a = {}
for i in range(1, num_alternative_sites + 1):
    for j in range(1, number_of_areas + 1):
        a[i, j] = 0
    for j in CoverableAreas[i]:
        a[i, j] = 1

# Create model
model = gp.Model("FireStationLocation")

# Decision variables: x_i = 1 if fire station built at site i
x = {}
for i in range(1, num_alternative_sites + 1):
    x[i] = model.addVar(vtype=gp.GRB.BINARY, name=f"x_{i}")

# Set objective: minimize total construction cost
obj = gp.quicksum(ConstructionCost[i] * x[i] for i in range(1, num_alternative_sites + 1))
model.setObjective(obj, gp.GRB.MINIMIZE)

# Coverage constraints based on importance weights
for j in range(1, number_of_areas + 1):
    weight = ImportanceWeight[j]
    
    # Skip library areas (5 and 6) as they have special constraints
    if j in library_areas:
        continue
    
    if weight_range_6_8[0] <= weight <= weight_range_6_8[1]:
        # Areas with weight 6-8: at least 2 coverage
        coverage = gp.quicksum(a[i, j] * x[i] for i in range(1, num_alternative_sites + 1))
        model.addConstr(coverage >= min_coverage_6_8, name=f"cov_weight6_8_area{j}")
    elif weight >= weight_threshold_9:
        # Areas with weight ≥9: at least 3 coverage
        coverage = gp.quicksum(a[i, j] * x[i] for i in range(1, num_alternative_sites + 1))
        model.addConstr(coverage >= min_coverage_9_up, name=f"cov_weight9_up_area{j}")
    else:
        # Other areas (weight < 6): at least 1 coverage
        coverage = gp.quicksum(a[i, j] * x[i] for i in range(1, num_alternative_sites + 1))
        model.addConstr(coverage >= min_coverage_other, name=f"cov_other_area{j}")

# Special constraints for library areas (5 and 6)
# Area 5: at least 3 coverage (since weight 7 needs 2, plus 1 extra)
coverage_area5 = gp.quicksum(a[i, 5] * x[i] for i in range(1, num_alternative_sites + 1))
model.addConstr(coverage_area5 >= 3, name="cov_library_area5")

# Area 6: at least 4 coverage (since weight 9 needs 3, plus 1 extra)
coverage_area6 = gp.quicksum(a[i, 6] * x[i] for i in range(1, num_alternative_sites + 1))
model.addConstr(coverage_area6 >= 4, name="cov_library_area6")

# Maximum number of fire stations constraint
total_stations = gp.quicksum(x[i] for i in range(1, num_alternative_sites + 1))
model.addConstr(total_stations <= max_fire_stations, name="max_stations")

# Solve the model
model.optimize()

# Print results
if model.status == gp.GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"Total cost: {model.objVal} (10,000 RMB)")
    
    print("\nSelected fire station sites:")
    selected_sites = []
    for i in range(1, num_alternative_sites + 1):
        if x[i].X > 0.5:
            selected_sites.append(i)
            print(f"  Site {i}: cost = {ConstructionCost[i]} (10,000 RMB)")
    
    print(f"\nNumber of stations built: {len(selected_sites)}")
    
    # Check coverage for each area
    print("\nCoverage analysis:")
    for j in range(1, number_of_areas + 1):
        coverage_count = sum(a[i, j] * x[i].X for i in range(1, num_alternative_sites + 1))
        required = 0
        weight = ImportanceWeight[j]
        
        if j in library_areas:
            if j == 5:
                required = 3
            elif j == 6:
                required = 4
        elif weight_range_6_8[0] <= weight <= weight_range_6_8[1]:
            required = min_coverage_6_8
        elif weight >= weight_threshold_9:
            required = min_coverage_9_up
        else:
            required = min_coverage_other
            
        status = "✓" if coverage_count >= required else "✗"
        print(f"  Area {j} (weight {weight}): {coverage_count} coverage, required {required} {status}")
    
    # Final answer: minimum total cost
    print(f"\nFinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found.")
    print("FinalAnswer=【None】")