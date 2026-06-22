import gurobipy as gp

# ====== 1. Parameter Definition ======
# Parameters from the Parameters List
J = 8  # number_of_candidate_sites
I = 10  # number_of_user_areas
B = 160  # total_construction_budget
max_sites = 5  # max_selected_sites
basic_cost = 0.5  # basic_population_service_cost
equip_cost = 1  # subway_equipment_cost
cost_reduction = 0.1  # subway_equipment_cost_reduction

# coverable_areas (Note: list index 0 is empty, stations are 1-8)
coverable_list = [[], [1, 2, 5], [2, 3, 6], [1, 4, 7], [3, 5, 6, 7],
                  [4, 6, 8, 9], [5, 7, 8, 10], [8, 9, 10], [9, 10]]

# base_station_construction_cost (Note: list index 0 is 0, costs are for stations 1-8)
c = [0, 27, 25, 25, 30, 30, 30, 25, 20]

# population (Note: list index 0 is 0, populations are for areas 1-10)
pop = [0, 4, 8, 5, 10, 12, 7, 9, 3, 6, 11]

# is_there_subway (Note: list index 0 is 0, subway indicators are for areas 1-10)
s = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1]

# Create coverage matrix a[i][j] = 1 if site j covers area i
a = [[0] * (J + 1) for _ in range(I + 1)]  # indices 1-based for clarity
for j in range(1, J + 1):
    for i in coverable_list[j]:
        a[i][j] = 1

# ====== 2. Create Model ======
model = gp.Model("BaseStationSelection")

# ====== 3. Decision Variables ======
x = {}  # x[j] = 1 if site j selected
for j in range(1, J + 1):
    x[j] = model.addVar(vtype=gp.GRB.BINARY, name=f"x_{j}")

y = {}  # y[i] = 1 if area i is covered
for i in range(1, I + 1):
    y[i] = model.addVar(vtype=gp.GRB.BINARY, name=f"y_{i}")

e = {}  # e[i] = 1 if subway equipment installed in area i
for i in range(1, I + 1):
    e[i] = model.addVar(vtype=gp.GRB.BINARY, name=f"e_{i}")

w = {}  # w[i] = population service cost for area i (in ten thousand yuan)
for i in range(1, I + 1):
    w[i] = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f"w_{i}")

# ====== 4. Objective Function ======
# Total cost = construction cost + population service cost + equipment cost
obj = gp.quicksum(c[j] * x[j] for j in range(1, J + 1))
obj += gp.quicksum(w[i] for i in range(1, I + 1))
obj += gp.quicksum(e[i] for i in range(1, I + 1))
model.setObjective(obj, gp.GRB.MINIMIZE)

# ====== 5. Constraints ======
# 5.1 Site-selection limit
model.addConstr(gp.quicksum(x[j] for j in range(1, J + 1)) <= max_sites, "max_sites")

# 5.2 Total budget constraint
model.addConstr(gp.quicksum(c[j] * x[j] for j in range(1, J + 1))
                + gp.quicksum(e[i] for i in range(1, I + 1)) <= B, "budget")

# 5.3 Coverage logic (lower bound): if covered (y_i=1), at least one covering site selected
for i in range(1, I + 1):
    model.addConstr(gp.quicksum(a[i][j] * x[j] for j in range(1, J + 1)) >= y[i], f"cover_lower_{i}")

# 5.4 Coverage logic (upper bound): if not covered (y_i=0), no covering site selected
# This is equivalent to: sum_j a_ij * x_j <= (sum_j a_ij) * y_i
for i in range(1, I + 1):
    total_coverable = sum(a[i][j] for j in range(1, J + 1))
    model.addConstr(gp.quicksum(a[i][j] * x[j] for j in range(1, J + 1))
                    <= total_coverable * y[i], f"cover_upper_{i}")

# 5.5 Global coverage requirement: each area must be covered
for i in range(1, I + 1):
    model.addConstr(gp.quicksum(a[i][j] * x[j] for j in range(1, J + 1)) >= 1, f"must_cover_{i}")

# 5.6 Interference constraint for areas 8,9,10: at most one covering base station
for i in [8, 9, 10]:
    model.addConstr(gp.quicksum(a[i][j] * x[j] for j in range(1, J + 1)) <= 1, f"interference_{i}")

# 5.7 Equipment only where subway exists
for i in range(1, I + 1):
    model.addConstr(e[i] <= s[i], f"subway_only_{i}")

# 5.8 Population service cost definition: w_i >= 0.5*pop_i*y_i - 0.1*pop_i*e_i
for i in range(1, I + 1):
    model.addConstr(w[i] >= basic_cost * pop[i] * y[i] - cost_reduction * pop[i] * e[i], f"pop_cost_{i}")

# ====== 6. Solve and Output Results ======
model.optimize()

# Check if solution is found
if model.status == gp.GRB.OPTIMAL:
    total_cost = model.ObjVal
    selected_sites = [j for j in range(1, J + 1) if x[j].X > 0.5]
    
    print("Optimal Solution Found")
    print(f"Selected base stations: {selected_sites}")
    print(f"Number of selected sites: {len(selected_sites)}")
    print(f"Total cost: {total_cost:.2f} (ten thousand yuan)")
    
    # Detailed cost breakdown
    constr_cost = sum(c[j] * x[j].X for j in range(1, J + 1))
    equip_cost_total = sum(e[i].X for i in range(1, I + 1))
    pop_cost_total = sum(w[i].X for i in range(1, I + 1))
    
    print(f"  Construction cost: {constr_cost:.2f}")
    print(f"  Equipment cost: {equip_cost_total:.2f}")
    print(f"  Population service cost: {pop_cost_total:.2f}")
    
    # Coverage status
    print("\nCoverage status:")
    for i in range(1, I + 1):
        covering_sites = [j for j in range(1, J + 1) if a[i][j] > 0.5 and x[j].X > 0.5]
        print(f"  Area {i}: covered by stations {covering_sites}")
    
    # Subway equipment installation
    subway_equipped = [i for i in range(1, I + 1) if e[i].X > 0.5]
    print(f"\nSubway equipment installed in areas: {subway_equipped}")
    
    # Final answer (total cost) as required
    print(f"FinalAnswer=【{total_cost:.2f}】")
    
else:
    print("No optimal solution found")
    print(f"FinalAnswer=【Infeasible】")