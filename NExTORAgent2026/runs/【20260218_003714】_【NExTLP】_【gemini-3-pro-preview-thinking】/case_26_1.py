import gurobipy as gp
from gurobipy import GRB

# 1. Define Parameters and Data Inputs
part_types = ['A', 'B', 'C']
worker_levels = [1, 2, 3, 4, 5, 6]
hours_per_week = 40

# Demand D_j
weekly_demand = {'A': 1940, 'B': 1000, 'C': 10060}

# Table C-7 Data: [Level, Number, Wage, Current_A, Current_B, Current_C]
# We extract Number (N_i) and Wage (W_i)
table_c7_data = [
    [1, 4, 15.0, 160, 0, 0],
    [2, 9, 14.5, 360, 0, 0],
    [3, 20, 13.0, 600, 200, 0],
    [4, 54, 12.0, 0, 160, 2000],
    [5, 102, 10.5, 0, 80, 4000],
    [6, 40, 9.75, 0, 0, 1600]
]
num_workers = {row[0]: row[1] for row in table_c7_data}  # N_i
hourly_wages = {row[0]: row[2] for row in table_c7_data}  # W_i

# Table Training Cost Data: [Level, Cost_A, Cost_B, Cost_C]
# Extract C_ij
table_training_data = [
    [1, 0, 10, 5],
    [2, 0, 20, 5],
    [3, 0, 0, 10],
    [4, 15, 0, 0],
    [5, 20, 0, 0],
    [6, 25, 20, 0]
]
training_costs = {}
for row in table_training_data:
    level = row[0]
    training_costs[(level, 'A')] = row[1]
    training_costs[(level, 'B')] = row[2]
    training_costs[(level, 'C')] = row[3]

# Table C-8 Efficiency Data: [Level, Eff_A, Eff_B, Eff_C]
# Extract E_ij
table_efficiency_data = [
    [1, 2.00, 1.20, 2.00],
    [2, 1.80, 1.08, 1.80],
    [3, 1.62, 2.50, 1.62],
    [4, 1.80, 2.16, 1.45],
    [5, 1.62, 1.93, 1.31],
    [6, 1.30, 1.74, 1.20]
]
efficiency = {}
for row in table_efficiency_data:
    level = row[0]
    efficiency[(level, 'A')] = row[1]
    efficiency[(level, 'B')] = row[2]
    efficiency[(level, 'C')] = row[3]

# 2. Create Model
model = gp.Model("WorkerScheduling")

# 3. Create Decision Variables
# h_{ij}: weekly working hours of level i workers on line j
h = model.addVars(worker_levels, part_types, vtype=GRB.CONTINUOUS, name="h", lb=0)

# k_{ij}: number of level i workers trained/assigned for line j
k = model.addVars(worker_levels, part_types, vtype=GRB.INTEGER, name="k", lb=0)

# 4. Set Objective Function
# Minimize Z = ∑ ∑ (w_i * h_{ij} + c_{ij} * k_{ij})
# Represents one week of salary plus one-time training costs.
objective_expr = gp.quicksum(
    hourly_wages[i] * h[i, j] + training_costs[(i, j)] * k[i, j]
    for i in worker_levels for j in part_types
)
model.setObjective(objective_expr, GRB.MINIMIZE)

# 5. Add Constraints

# Constraint 1: Time capacity per level
# ∑_{j} h_{ij} ≤ 40 * N_i for each level i
for i in worker_levels:
    model.addConstr(
        gp.quicksum(h[i, j] for j in part_types) <= hours_per_week * num_workers[i],
        name=f"Time_Capacity_Level_{i}"
    )

# Constraint 2: Demand satisfaction
# ∑_{i} E_{ij} * h_{ij} ≥ D_j for each part j
for j in part_types:
    model.addConstr(
        gp.quicksum(efficiency[(i, j)] * h[i, j] for i in worker_levels) >= weekly_demand[j],
        name=f"Demand_Satisfaction_{j}"
    )

# Constraint 3: Training capacity link
# h_{ij} ≤ 40 * k_{ij}
# This ensures that if hours are assigned, enough workers are trained/assigned.
for i in worker_levels:
    for j in part_types:
        model.addConstr(
            h[i, j] <= hours_per_week * k[i, j],
            name=f"Link_Training_Hours_{i}_{j}"
        )

# Constraint 4: Training upper bound
# k_{ij} ≤ N_i
# The number of workers of level i assigned to line j cannot exceed total workers of level i.
for i in worker_levels:
    for j in part_types:
        model.addConstr(
            k[i, j] <= num_workers[i],
            name=f"Max_Workers_Assigned_{i}_{j}"
        )

# 6. Solve and Print Results
model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"Optimal Total Expenditure: {model.ObjVal:.2f}")
    print("\nDetailed Schedule:")
    for i in worker_levels:
        for j in part_types:
            if h[i, j].x > 0.001 or k[i, j].x > 0.5:
                print(f"  Level {i} on Line {j}: Hours={h[i, j].x:.2f}, Assigned Workers={k[i, j].x}")
else:
    print("Optimization was not successful.")

print(f"FinalAnswer=【{model.ObjVal}】")