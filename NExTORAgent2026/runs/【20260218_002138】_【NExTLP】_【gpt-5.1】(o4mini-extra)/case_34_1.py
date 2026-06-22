import gurobipy as gp
from gurobipy import GRB

# ===============================
# 1. Parameters from Parameters List
# ===============================

year_report = 2023
num_main_roads = [4, 5]
action_plan_years = 3
industrial_areas = [3, 7]
commercial_areas = [2, 6]
residential_areas = [1, 4, 5]
roads_exist = [[4, 5], [5, 6]]
total_persons = 21100
total_mileage = 399250
ferry_arc = (2, 6)
ferry_capacity = 2000
car_occupancy = 1
q_values = [1, 2, 3]

Table_1_map = [
    ['Mountain', 'Mountain', 'Mountain', 'Mountain'],
    ['Mountain', 'Mountain', 'Mountain', 'Mountain'],
    [1, 2, 3, 4],
    ['Lake', 'Lake (to build a ferry)', 'Lake', 'Road 4-5'],
    ['Lake', 'Lake (to build a ferry)', 'Lake', 'Road 4-5'],
    [7, 6, 'Road 5-6', 5],
    ['Mountain', 'Mountain', 'Mountain', 'Mountain']
]

Table_2_distances = {
    '1-2': 3.5,
    '2-3': 3.0,
    '3-4': 5.0,
    '4-5': 25.0,
    '5-6': 4.0,
    '6-7': 2.5,
    '2-6': 'To be established'  # ferry distance; not needed for reduction since we model relative to total_mileage
}

Table_3_C35 = {
    '1': {'1': 0, '2': 900, '3': 750, '4': 40, '5': 10, '6': 600, '7': 550, 'total': 2850},
    '4': {'1': 100, '2': 2000, '3': 1100, '4': 0, '5': 150, '6': 1400, '7': 1250, 'total': 6000},
    '5': {'1': 110, '2': 4000, '3': 2200, '4': 200, '5': 0, '6': 3300, '7': 2440, 'total': 12250}
}

# ===============================
# 2. Derived sets and data
# ===============================

# Nodes
V = [1, 2, 3, 4, 5, 6, 7]

# Directed arcs E (road arcs + ferry arcs)
E = [
    (1, 2), (2, 1),
    (2, 3), (3, 2),
    (3, 4), (4, 3),
    (4, 5), (5, 4),
    (5, 6), (6, 5),
    (6, 7), (7, 6),
    (2, 6), (6, 2)
]

# Distances C_ij, symmetric, using given values; for 2-6 ferry we must choose a distance.
# The original statement doesn't give a numeric value for 2-6; to keep the model
# consistent and computable, we set a reasonable distance (e.g., same order of magnitude
# as neighboring arcs). This distance affects the optimal flows but not the structure.
# You could change this value as needed.
C = {}
C[(1, 2)] = C[(2, 1)] = Table_2_distances['1-2']
C[(2, 3)] = C[(3, 2)] = Table_2_distances['2-3']
C[(3, 4)] = C[(4, 3)] = Table_2_distances['3-4']
C[(4, 5)] = C[(5, 4)] = Table_2_distances['4-5']
C[(5, 6)] = C[(6, 5)] = Table_2_distances['5-6']
C[(6, 7)] = C[(7, 6)] = Table_2_distances['6-7']

# Set ferry distance: here we assume 2 km (you can adjust if a specific value is known)
ferry_distance = 2.0
C[(2, 6)] = C[(6, 2)] = ferry_distance

# Origins mapping: q=1 -> node 1, q=2 -> node 4, q=3 -> node 5
origin_of_q = {1: 1, 2: 4, 3: 5}

# Build demand b_{qk}
b = {}
for q in q_values:
    b[q] = {}
    for k in V:
        b[q][k] = 0

# q=1 corresponds to residential node 1
for k_str, val in Table_3_C35['1'].items():
    if k_str == 'total':
        continue
    k = int(k_str)
    b[1][k] = val

# q=2 corresponds to residential node 4
for k_str, val in Table_3_C35['4'].items():
    if k_str == 'total':
        continue
    k = int(k_str)
    b[2][k] = val

# q=3 corresponds to residential node 5
for k_str, val in Table_3_C35['5'].items():
    if k_str == 'total':
        continue
    k = int(k_str)
    b[3][k] = val

# Supply/demand s_{q,i}
s = {}
for q in q_values:
    s[q] = {}
    origin = origin_of_q[q]
    total_out = sum(b[q][k] for k in V)
    for i in V:
        if i == origin:
            s[q][i] = total_out
        else:
            s[q][i] = -b[q][i]

# ===============================
# 3. Create model and decision variables
# ===============================

model = gp.Model("BinhaiHarbor_Ferry_Optimization")

# Flow variables x_{q,i,j} >= 0
x = model.addVars(
    q_values, V, V,
    vtype=GRB.CONTINUOUS,
    lb=0.0,
    name="x"
)

# Reduction variable R >= 0
R = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="R")

model.update()

# ===============================
# 4. Objective: Maximize mileage reduction R
# ===============================

model.setObjective(R, GRB.MAXIMIZE)

# ===============================
# 5. Constraints
# ===============================

# 5.1 Flow conservation for each q, i
for q in q_values:
    for i in V:
        out_flow = gp.quicksum(x[q, i, j] for (ii, j) in E if ii == i)
        in_flow = gp.quicksum(x[q, j, i] for (j, jj) in E if jj == i)
        model.addConstr(out_flow - in_flow == s[q][i], name=f"flow_conserv_q{q}_i{i}")

# 5.2 Ferry capacity constraints for arcs (2,6) and (6,2)
model.addConstr(
    gp.quicksum(x[q, 2, 6] for q in q_values) <= ferry_capacity,
    name="ferry_cap_2_6"
)
model.addConstr(
    gp.quicksum(x[q, 6, 2] for q in q_values) <= ferry_capacity,
    name="ferry_cap_6_2"
)

# 5.3 Total mileage and reduction relationship:
#    R + sum_{q,(i,j) in E} C_ij * x_{qij} = total_mileage
total_new_mileage = gp.quicksum(C[(i, j)] * x[q, i, j] for q in q_values for (i, j) in E)
model.addConstr(
    R + total_new_mileage == total_mileage,
    name="reduction_relation"
)

# ===============================
# 6. Optimize
# ===============================

model.optimize()

# ===============================
# 7. Print results
# ===============================

if model.Status == GRB.OPTIMAL:
    print(f"Optimal objective (R, mileage reduction) = {R.X}")
    new_mileage_value = total_mileage - R.X
    print(f"New total mileage after ferry opens = {new_mileage_value}")
    print("Non-zero flows x[q,i,j]:")
    for q in q_values:
        for (i, j) in E:
            val = x[q, i, j].X
            if val > 1e-6:
                print(f"x[{q},{i},{j}] = {val}")
    final_answer = R.X
else:
    print("Model did not reach optimality.")
    final_answer = None

# Final answer output as required
print(f"FinalAnswer=【{final_answer}】")