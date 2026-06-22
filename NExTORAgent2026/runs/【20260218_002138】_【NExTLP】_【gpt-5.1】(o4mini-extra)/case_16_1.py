import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters (from Parameters List)
# =========================
number_of_candidate_sites = 8
number_of_user_areas = 10
total_construction_budget = 160
max_selected_sites = 5
average_construction_period_years = 1  # not directly used in model
basic_population_service_cost = 0.5
subway_equipment_cost = 1
subway_equipment_cost_reduction = 0.1

# Index 0 is dummy to keep 1-based indexing consistent with the statement
coverable_areas = [
    [],              # 0 - dummy
    [1, 2, 5],       # site 1
    [2, 3, 6],       # site 2
    [1, 4, 7],       # site 3
    [3, 5, 6, 7],    # site 4
    [4, 6, 8, 9],    # site 5
    [5, 7, 8, 10],   # site 6
    [8, 9, 10],      # site 7
    [9, 10]          # site 8
]

base_station_construction_cost = [
    0,   # dummy
    27,  # c1
    25,  # c2
    25,  # c3
    30,  # c4
    30,  # c5
    30,  # c6
    25,  # c7
    20   # c8
]

population = [
    0,   # dummy
    4,   # area 1
    8,   # area 2
    5,   # area 3
    10,  # area 4
    12,  # area 5
    7,   # area 6
    9,   # area 7
    3,   # area 8
    6,   # area 9
    11   # area 10
]

is_there_subway = [
    0,  # dummy
    0,  # area 1
    0,  # area 2
    0,  # area 3
    1,  # area 4
    1,  # area 5
    0,  # area 6
    0,  # area 7
    0,  # area 8
    0,  # area 9
    1   # area 10
]

# =========================
# 2. Derived data: coverage matrix a_ij
# =========================
# a[i][j] = 1 if site j can cover area i, else 0
a = [[0] * (number_of_candidate_sites + 1) for _ in range(number_of_user_areas + 1)]
for j in range(1, number_of_candidate_sites + 1):
    for i in coverable_areas[j]:
        a[i][j] = 1

# Precompute sum_j a_ij for upper-bound coverage constraints
sum_a_i = [0] * (number_of_user_areas + 1)
for i in range(1, number_of_user_areas + 1):
    sum_a_i[i] = sum(a[i][j] for j in range(1, number_of_candidate_sites + 1))

# =========================
# 3. Create model
# =========================
model = gp.Model("BaseStation_Layout_Optimization")

# =========================
# 4. Decision variables
# =========================
# x_j: 1 if site j is selected
x = model.addVars(
    range(1, number_of_candidate_sites + 1),
    vtype=GRB.BINARY,
    name="x"
)

# y_i: 1 if user area i is covered
y = model.addVars(
    range(1, number_of_user_areas + 1),
    vtype=GRB.BINARY,
    name="y"
)

# e_i: 1 if subway equipment installed in area i
e = model.addVars(
    range(1, number_of_user_areas + 1),
    vtype=GRB.BINARY,
    name="e"
)

# =========================
# 5. Objective function: Minimize total cost
# Z = sum_j c_j x_j
#   + sum_i (0.5 * pop_i * y_i - 0.1 * pop_i * e_i)
#   + sum_i e_i
# =========================
construction_cost = gp.quicksum(
    base_station_construction_cost[j] * x[j]
    for j in range(1, number_of_candidate_sites + 1)
)

population_service_cost = gp.quicksum(
    basic_population_service_cost * population[i] * y[i]
    for i in range(1, number_of_user_areas + 1)
)

subway_reduction = gp.quicksum(
    subway_equipment_cost_reduction * population[i] * e[i]
    for i in range(1, number_of_user_areas + 1)
)

subway_equipment_total = gp.quicksum(
    subway_equipment_cost * e[i]
    for i in range(1, number_of_user_areas + 1)
)

model.setObjective(
    construction_cost + population_service_cost - subway_reduction + subway_equipment_total,
    GRB.MINIMIZE
)

# =========================
# 6. Constraints
# =========================

# 6.1 Site-selection limit: sum_j x_j <= max_selected_sites
model.addConstr(
    gp.quicksum(x[j] for j in range(1, number_of_candidate_sites + 1)) <= max_selected_sites,
    name="SiteSelectionLimit"
)

# 6.2 Total budget: sum_j c_j x_j + sum_i e_i <= total_construction_budget
model.addConstr(
    construction_cost + subway_equipment_total <= total_construction_budget,
    name="TotalBudget"
)

# 6.3 Coverage logic (lower bound): sum_j a_ij x_j >= y_i, for all i
for i in range(1, number_of_user_areas + 1):
    model.addConstr(
        gp.quicksum(a[i][j] * x[j] for j in range(1, number_of_candidate_sites + 1)) >= y[i],
        name=f"CoverageLower_{i}"
    )

# 6.4 Coverage logic (upper bound): sum_j a_ij x_j <= (sum_j a_ij) * y_i, for all i
for i in range(1, number_of_user_areas + 1):
    if sum_a_i[i] > 0:
        model.addConstr(
            gp.quicksum(a[i][j] * x[j] for j in range(1, number_of_candidate_sites + 1)) <=
            sum_a_i[i] * y[i],
            name=f"CoverageUpper_{i}"
        )
    else:
        # no site can cover this area -> model would be infeasible; but we keep structure consistent
        model.addConstr(y[i] == 0, name=f"CoverageImpossible_{i}")

# 6.5 Global coverage requirement: sum_j a_ij x_j >= 1, for all i
for i in range(1, number_of_user_areas + 1):
    model.addConstr(
        gp.quicksum(a[i][j] * x[j] for j in range(1, number_of_candidate_sites + 1)) >= 1,
        name=f"GlobalCoverage_{i}"
    )

# 6.6 Interference in areas 8, 9, 10: sum_j a_ij x_j <= 1
for i in [8, 9, 10]:
    model.addConstr(
        gp.quicksum(a[i][j] * x[j] for j in range(1, number_of_candidate_sites + 1)) <= 1,
        name=f"Interference_{i}"
    )

# 6.7 Equipment only where subway exists: e_i <= s_i
for i in range(1, number_of_user_areas + 1):
    model.addConstr(
        e[i] <= is_there_subway[i],
        name=f"SubwayEquipFeasible_{i}"
    )

# =========================
# 7. Solve the model
# =========================
model.Params.OutputFlag = 0  # silence solver output; set to 1 for details
model.optimize()

if model.status == GRB.OPTIMAL:
    print("Optimal objective value (total cost):", model.objVal)

    print("\nSelected base stations (x_j = 1):")
    for j in range(1, number_of_candidate_sites + 1):
        if x[j].X > 0.5:
            print(f"  Site {j} selected")

    print("\nArea coverage indicators (y_i):")
    for i in range(1, number_of_user_areas + 1):
        print(f"  Area {i} covered: {int(round(y[i].X))}")

    print("\nSubway equipment installation (e_i):")
    for i in range(1, number_of_user_areas + 1):
        if e[i].X > 0.5:
            print(f"  Subway equipment installed in area {i}")

    # The question asks: "It is required that all areas be covered and the total cost be provided."
    # FinalAnswer is the optimal total cost.
    final_answer = model.objVal
else:
    # If not optimal, we set FinalAnswer to None (or a sentinel); here we use -1 to keep it numeric
    final_answer = -1
    print("Model did not reach optimality. Status code:", model.status)

print(f"FinalAnswer=【{final_answer}】")