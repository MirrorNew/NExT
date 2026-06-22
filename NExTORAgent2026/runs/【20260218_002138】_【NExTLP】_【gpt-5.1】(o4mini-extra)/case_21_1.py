import gurobipy as gp
from gurobipy import GRB

# ==============================
# 1. Parameters (given)
# ==============================
number_of_areas = 12
weight_range_6_8 = (6, 8)
min_coverage_6_8 = 2
weight_threshold_9 = 9
min_coverage_9_up = 3
min_coverage_other = 1
max_fire_stations = 6
library_areas = [5, 6]
additional_coverage_for_library_areas = 1
coverage_time_limit = 5  # not directly used; coverage already encoded
num_alternative_sites = 8

# Index 0 is None to match 1-based indexing in the description
CoverableAreas = [
    None,
    [1, 2, 5, 6, 7, 8, 11],   # site 1
    [2, 3, 5, 6, 7],          # site 2
    [1, 4, 5, 6, 7, 8],       # site 3
    [3, 4, 5, 6, 7],          # site 4
    [6, 7, 10, 11],           # site 5
    [4, 5, 8, 11, 12],        # site 6
    [2, 9, 11, 12],           # site 7
    [5, 10, 12]               # site 8
]

ConstructionCost = [
    None,
    90,  # site 1
    70,  # site 2
    85,  # site 3
    65,  # site 4
    60,  # site 5
    80,  # site 6
    60,  # site 7
    50   # site 8
]

ImportanceWeight = [
    None,
    4,   # area 1
    8,   # area 2
    5,   # area 3
    10,  # area 4
    7,   # area 5
    9,   # area 6
    6,   # area 7
    6,   # area 8
    3,   # area 9
    5,   # area 10
    8,   # area 11
    4    # area 12
]

# ==============================
# 2. Derived data: coverage matrix a_{ij}
# ==============================
sites = range(1, num_alternative_sites + 1)
areas = range(1, number_of_areas + 1)

# a[i][j] = 1 if site i covers area j, else 0
a = {(i, j): 0 for i in sites for j in areas}
for i in sites:
    for j in CoverableAreas[i]:
        a[(i, j)] = 1

# ==============================
# 3. Create model
# ==============================
model = gp.Model("Xunan_FireStation_RobustCoverage")

# ==============================
# 4. Decision variables
# ==============================
# x_i = 1 if a station is built at site i
x = model.addVars(sites, vtype=GRB.BINARY, name="x")

# ==============================
# 5. Objective: minimize total construction cost
# ==============================
model.setObjective(
    gp.quicksum(ConstructionCost[i] * x[i] for i in sites),
    GRB.MINIMIZE
)

# ==============================
# 6. Constraints
# ==============================

# Helper: classify areas according to importance weight
areas_weight_6_8 = [j for j in areas if weight_range_6_8[0] <= ImportanceWeight[j] <= weight_range_6_8[1]]
areas_weight_9_up = [j for j in areas if ImportanceWeight[j] >= weight_threshold_9]
areas_other = [j for j in areas if ImportanceWeight[j] < weight_range_6_8[0]]

# 6.1 Coverage_weight6_8: sum_i a_ij x_i >= 2 for 6 <= w_j <= 8
for j in areas_weight_6_8:
    model.addConstr(
        gp.quicksum(a[(i, j)] * x[i] for i in sites) >= min_coverage_6_8,
        name=f"Coverage_weight6_8_area{j}"
    )

# 6.2 Coverage_weight≥9: sum_i a_ij x_i >= 3 for w_j >= 9
for j in areas_weight_9_up:
    model.addConstr(
        gp.quicksum(a[(i, j)] * x[i] for i in sites) >= min_coverage_9_up,
        name=f"Coverage_weight9up_area{j}"
    )

# 6.3 Coverage_others: sum_i a_ij x_i >= 1 for w_j < 6
for j in areas_other:
    model.addConstr(
        gp.quicksum(a[(i, j)] * x[i] for i in sites) >= min_coverage_other,
        name=f"Coverage_other_area{j}"
    )

# 6.4 Library_extra_area5 and area6:
# base requirement already imposed by the above three groups.
# We must add +1 extra coverage for library areas.
for j in library_areas:
    base_req = None
    wj = ImportanceWeight[j]
    if weight_range_6_8[0] <= wj <= weight_range_6_8[1]:
        base_req = min_coverage_6_8
    elif wj >= weight_threshold_9:
        base_req = min_coverage_9_up
    else:
        base_req = min_coverage_other

    total_required = base_req + additional_coverage_for_library_areas

    model.addConstr(
        gp.quicksum(a[(i, j)] * x[i] for i in sites) >= total_required,
        name=f"Library_extra_area{j}"
    )

# 6.5 Max_number_of_stations: sum_i x_i <= 6
model.addConstr(
    gp.quicksum(x[i] for i in sites) <= max_fire_stations,
    name="Max_number_of_stations"
)

# ==============================
# 7. Solve model
# ==============================
model.Params.OutputFlag = 0  # turn off Gurobi output; set to 1 to see log
model.optimize()

# ==============================
# 8. Print results
# ==============================
if model.status == GRB.OPTIMAL:
    print("Optimal solution found.")
    print(f"Minimum total construction cost (10,000 RMB): {model.ObjVal:.0f}")
    print("Build stations at sites:")
    for i in sites:
        if x[i].X > 0.5:
            print(f"  Site {i} (cost {ConstructionCost[i]})")
else:
    print(f"Optimization ended with status {model.status}")

# Final answer is the minimum total construction cost
final_answer_value = model.ObjVal if model.status == GRB.OPTIMAL else float("nan")
print(f"FinalAnswer=【{final_answer_value}】")