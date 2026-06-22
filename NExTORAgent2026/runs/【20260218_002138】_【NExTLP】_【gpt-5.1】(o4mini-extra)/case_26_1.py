import gurobipy as gp
from gurobipy import GRB

# ========= 1. Parameters from the given Parameters List =========

part_types = ['A', 'B', 'C']
worker_levels = [1, 2, 3, 4, 5, 6]
hours_per_week = 40
required_levels_A = [1, 2, 3]  # (Not specifically used in the model, but kept as given.)

weekly_demand = {'A': 1940, 'B': 1000, 'C': 10060}

# Table_1_C7: [level, number_of_workers, wage, A_hours, B_hours, C_hours]
Table_1_C7 = [
    [1, 4,   15.0, 160,   0,    0],
    [2, 9,   14.5, 360,   0,    0],
    [3, 20,  13.0, 600, 200,    0],
    [4, 54,  12.0,   0, 160, 2000],
    [5, 102, 10.5,   0,  80, 4000],
    [6, 40,  9.75,   0,   0, 1600],
]

# Table_2_TrainingCost: [level, train_cost_A, train_cost_B, train_cost_C]
Table_2_TrainingCost = [
    [1, 0,  10,  5],
    [2, 0,  20,  5],
    [3, 0,   0, 10],
    [4, 15,  0,  0],
    [5, 20,  0,  0],
    [6, 25, 20,  0],
]

# Table_3_C8: [level, rate_A, rate_B, rate_C]
Table_3_C8 = [
    [1, 2.0, 1.2, 2.0],
    [2, 1.8, 1.08, 1.8],
    [3, 1.62, 2.5, 1.62],
    [4, 1.8, 2.16, 1.45],
    [5, 1.62, 1.93, 1.31],
    [6, 1.3, 1.74, 1.2],
]

# ---- Derived parameter dictionaries ----

# Number of workers N_i and wages w_i
N = {}     # N[i] = number of workers of level i
wage = {}  # wage[i] = wage of level i

for row in Table_1_C7:
    level, num_workers, w_i = row[0], row[1], row[2]
    N[level] = num_workers
    wage[level] = w_i

# Training cost c_{ij}
train_cost = {}  # train_cost[i][j]
for row in Table_2_TrainingCost:
    level, cA, cB, cC = row
    train_cost[level] = {
        'A': cA,
        'B': cB,
        'C': cC
    }

# Production rate r_{ij}
rate = {}  # rate[i][j]
for row in Table_3_C8:
    level, rA, rB, rC = row
    rate[level] = {
        'A': rA,
        'B': rB,
        'C': rC
    }

# ========= 2. Create model =========

model = gp.Model("Hailong_Auto_Parts_Worker_Scheduling")

# ========= 3. Decision variables =========
# h_{ij}: total weekly working hours of level i workers on line j (continuous)
# k_{ij}: number of level i workers trained for line j (integer)

h = model.addVars(
    worker_levels,
    part_types,
    vtype=GRB.CONTINUOUS,
    name="h"
)

k = model.addVars(
    worker_levels,
    part_types,
    vtype=GRB.INTEGER,
    lb=0.0,
    name="k"
)

# ========= 4. Objective function =========
# Minimize Z = sum_{i,j} (w_i * h_{ij} + c_{ij} * k_{ij})

model.setObjective(
    gp.quicksum(
        wage[i] * h[i, j] + train_cost[i][j] * k[i, j]
        for i in worker_levels for j in part_types
    ),
    GRB.MINIMIZE
)

# ========= 5. Constraints =========

# (1) Time capacity per level: sum_j h_{ij} <= hours_per_week * N_i
for i in worker_levels:
    model.addConstr(
        gp.quicksum(h[i, j] for j in part_types) <= hours_per_week * N[i],
        name=f"TimeCapacity_level_{i}"
    )

# (2) Demand satisfaction: sum_i r_{ij} * h_{ij} >= D_j
for j in part_types:
    model.addConstr(
        gp.quicksum(rate[i][j] * h[i, j] for i in worker_levels) >= weekly_demand[j],
        name=f"Demand_{j}"
    )

# (3) Training capacity link: h_{ij} <= hours_per_week * k_{ij}
for i in worker_levels:
    for j in part_types:
        model.addConstr(
            h[i, j] <= hours_per_week * k[i, j],
            name=f"TrainingCapLink_{i}_{j}"
        )

# (4) Training upper bound: k_{ij} <= N_i
for i in worker_levels:
    for j in part_types:
        model.addConstr(
            k[i, j] <= N[i],
            name=f"TrainingUpperBound_{i}_{j}"
        )

# Nonnegativity for h_{ij} is already guaranteed by default lb=0 for continuous vars.
# k_{ij} are integer with lb=0 already set.

# ========= 6. Solve the model =========

model.optimize()

# ========= 7. Print results =========

if model.status == GRB.OPTIMAL:
    total_cost = model.objVal
    print("Optimal total weekly salary + training cost:", total_cost)

    print("\nWork hours h_{ij} (only nonzero):")
    for i in worker_levels:
        for j in part_types:
            val = h[i, j].X
            if val > 1e-6:
                print(f"Level {i}, Line {j}: {val:.2f} hours")

    print("\nTrained workers k_{ij} (only nonzero):")
    for i in worker_levels:
        for j in part_types:
            val = k[i, j].X
            if val > 1e-6:
                print(f"Level {i}, Line {j}: {val:.2f} workers")

    # FinalAnswer is the total cost (objective value)
    print(f"FinalAnswer=【{total_cost}】")
else:
    # If the model is not optimal, still output something for FinalAnswer
    print("Model did not solve to optimality. Status code:", model.status)
    print("FinalAnswer=【None】")