import gurobipy as gp
from gurobipy import GRB

# =============================
# 1. Parameters from Parameters List
# =============================

num_candidate_sites = 8
num_user_areas = 10
max_selected_sites = 5
total_budget = 160.0
average_construction_period_years = 1.0  # not used directly
base_population_service_cost_per_person_yuan = 0.5
subway_equipment_cost_ten_thousand_yuan = 1.0
subway_equipment_reduction_per_person_yuan = 0.1
interference_areas = [8, 9, 10]

Table_1_BaseStationCoverageRelationship_base_station_numbers = [1, 2, 3, 4, 5, 6, 7, 8]
Table_1_BaseStationCoverageRelationship_coverable_user_areas = [
    [1, 2, 5],       # station 1
    [2, 3, 6],       # station 2
    [1, 4, 7],       # station 3
    [3, 5, 6, 7],    # station 4
    [4, 6, 8, 9],    # station 5
    [5, 7, 8, 10],   # station 6
    [8, 9, 10],      # station 7
    [9, 10],         # station 8
]
Table_1_BaseStationCoverageRelationship_construction_costs = [27.0, 25.0, 25.0, 30.0, 30.0, 30.0, 25.0, 20.0]

Table_2_UserAreaSituation_area_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Table_2_UserAreaSituation_population_ten_thousand_people = [4.0, 8.0, 5.0, 10.0, 12.0, 7.0, 9.0, 3.0, 6.0, 11.0]
Table_2_UserAreaSituation_has_subway = [0, 0, 0, 1, 1, 0, 0, 0, 0, 1]

# =============================
# 2. Derived sets / parameters
# =============================

# Stations and areas
stations = Table_1_BaseStationCoverageRelationship_base_station_numbers  # [1..8]
areas = Table_2_UserAreaSituation_area_numbers                           # [1..10]

# Construction cost per station
f = {stations[i]: Table_1_BaseStationCoverageRelationship_construction_costs[i]
     for i in range(num_candidate_sites)}

# Population per area (ten thousand people)
P = {areas[j]: Table_2_UserAreaSituation_population_ten_thousand_people[j]
     for j in range(num_user_areas)}

# Subway indicator I_subway(j)
I_subway = {areas[j]: Table_2_UserAreaSituation_has_subway[j]
            for j in range(num_user_areas)}

# Coverage sets C(j): stations that can cover area j
C = {j: [] for j in areas}
for idx, i in enumerate(stations):
    for j in Table_1_BaseStationCoverageRelationship_coverable_user_areas[idx]:
        C[j].append(i)

# Subway areas (where I_subway(j)=1)
subway_areas = [j for j in areas if I_subway[j] == 1]  # [4, 5, 10]

# =============================
# 3. Build model and variables
# =============================

model = gp.Model("5G_BaseStation_Layout")

# x_i: 1 if station i is built
x = model.addVars(stations, vtype=GRB.BINARY, name="x")

# a_{ij}: 1 if area j is assigned to station i that can cover j
a = {}
for j in areas:
    for i in C[j]:
        a[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"a_{i}_{j}")

# y_j: 1 if area j is covered (will be fixed to 1)
y = model.addVars(areas, vtype=GRB.BINARY, name="y")

# z_j: 1 if subway coverage equipment is installed in subway area j
z = model.addVars(subway_areas, vtype=GRB.BINARY, name="z")

# k_j: number of stations serving area j (integer)
k = model.addVars(areas, vtype=GRB.INTEGER, name="k")

model.update()

# =============================
# 4. Constraints
# =============================

# Station selection limit: sum_i x_i <= 5
model.addConstr(gp.quicksum(x[i] for i in stations) <= max_selected_sites,
                name="station_selection_limit")

# Assignment only if station is built: a_{ij} <= x_i
for j in areas:
    for i in C[j]:
        model.addConstr(a[(i, j)] <= x[i], name=f"assign_only_if_built_{i}_{j}")

# Full coverage of each area: sum_{i in C(j)} a_{ij} >= 1
for j in areas:
    model.addConstr(gp.quicksum(a[(i, j)] for i in C[j]) >= 1,
                    name=f"full_coverage_area_{j}")

# Coverage status definition: y_j <= sum_{i in C(j)} a_{ij}, and y_j = 1
for j in areas:
    model.addConstr(y[j] <= gp.quicksum(a[(i, j)] for i in C[j]),
                    name=f"coverage_status_upper_{j}")
    model.addConstr(y[j] == 1, name=f"coverage_status_fixed_{j}")

# Interference constraints for areas 8, 9, 10: sum_{i in C(j)} a_{ij} <= 1
for j in interference_areas:
    model.addConstr(gp.quicksum(a[(i, j)] for i in C[j]) <= 1,
                    name=f"interference_area_{j}")

# Subway equipment linked to coverage in subway areas: z_j <= y_j
for j in subway_areas:
    model.addConstr(z[j] <= y[j], name=f"subway_equipment_link_{j}")

# Definition of k_j: k_j = sum_{i in C(j)} a_{ij}
for j in areas:
    model.addConstr(k[j] == gp.quicksum(a[(i, j)] for i in C[j]),
                    name=f"definition_k_{j}")

# =============================
# 5. Objective function
# =============================

# Construction cost: sum_i f_i * x_i
construction_cost = gp.quicksum(f[i] * x[i] for i in stations)

# Subway equipment cost: 1 (ten thousand yuan) per installed equipment
subway_equipment_cost = gp.quicksum(subway_equipment_cost_ten_thousand_yuan * z[j]
                                    for j in subway_areas)

# Population service cost:
# Population_Service_Cost = sum_j P_j * (0.5 - 0.1 * z_j * I_subway(j))
population_cost_constant = gp.quicksum(
    P[j] * base_population_service_cost_per_person_yuan for j in areas
)
population_cost_reduction = gp.quicksum(
    P[j] * subway_equipment_reduction_per_person_yuan * z[j] * I_subway[j]
    for j in subway_areas
)
population_service_cost = population_cost_constant - population_cost_reduction

total_cost = construction_cost + subway_equipment_cost + population_service_cost

# Budget constraint: Total_Cost <= 160
model.addConstr(total_cost <= total_budget, name="budget_constraint")

# Set objective: minimize total_cost
model.setObjective(total_cost, GRB.MINIMIZE)

# =============================
# 6. Solve model
# =============================

model.optimize()

# =============================
# 7. Print results and FinalAnswer
# =============================

if model.status == GRB.OPTIMAL:
    print("\nOptimal solution found.")
    print(f"Minimum total cost: {model.objVal:.4f}")

    const_cost_val = construction_cost.getValue()
    subway_equipment_cost_val = subway_equipment_cost.getValue()
    population_cost_val = population_service_cost.getValue()

    print(f"  Construction cost: {const_cost_val:.4f}")
    print(f"  Subway equipment cost: {subway_equipment_cost_val:.4f}")
    print(f"  Population service cost: {population_cost_val:.4f}")

    print("\nSelected base stations (x_i = 1):")
    for i in stations:
        if x[i].X > 0.5:
            print(f"  Station {i} (construction cost {f[i]})")

    print("\nSubway equipment installation (z_j = 1):")
    for j in subway_areas:
        if z[j].X > 0.5:
            print(f"  Area {j} (population {P[j]})")

    print("\nArea assignments a_{ij} = 1:")
    for j in areas:
        assigned_stations = [i for i in C[j] if a[(i, j)].X > 0.5]
        print(f"  Area {j}: served by stations {assigned_stations}, k_j = {k[j].X}")

    # The question asks: "It is required that all areas be covered and the total cost be provided."
    # So FinalAnswer is the minimum total cost.
    FinalAnswer = model.objVal
    print(f"FinalAnswer=【{FinalAnswer}】")
else:
    print(f"\nOptimization ended with status {model.status}. No optimal solution reported.")
    # In case of non-optimality, we still output something for FinalAnswer (e.g., NaN or None)
    FinalAnswer = float('nan')
    print(f"FinalAnswer=【{FinalAnswer}】")