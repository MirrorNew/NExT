import gurobipy as gp
from gurobipy import GRB

# Parameters from the list
part_types = ['A', 'B', 'C']
worker_levels = [1, 2, 3, 4, 5, 6]
hours_per_week = 40

# Weekly demand for accessories A, B, and C
weekly_demand = {'A': 1940, 'B': 1000, 'C': 10060}

# Worker information from Table C-7: Number of workers (N_i) and hourly wage (w_i)
# Table_1_C7: [Level, N_i, w_i, CurrentLineAHours, CurrentLineBHours, CurrentLineCHours]
N = [4, 9, 20, 54, 102, 40]
W = [15.0, 14.5, 13.0, 12.0, 10.5, 9.75]

# Training cost per person for each level and production line from Table_2_TrainingCost
# Rows: Worker levels 1-6; Columns: Line A, Line B, Line C
C_cost = [
    [0, 10, 5],    # Level 1
    [0, 20, 5],    # Level 2
    [0, 0, 10],    # Level 3
    [15, 0, 0],    # Level 4
    [20, 0, 0],    # Level 5
    [25, 20, 0]    # Level 6
]

# Work efficiency (productivity) in pieces per hour from Table_3_C8
# Rows: Worker levels 1-6; Columns: Line A, Line B, Line C
R_prod = [
    [2.0, 1.2, 2.0],   # Level 1
    [1.8, 1.08, 1.8],  # Level 2
    [1.62, 2.5, 1.62], # Level 3
    [1.8, 2.16, 1.45], # Level 4
    [1.62, 1.93, 1.31],# Level 5
    [1.3, 1.74, 1.2]   # Level 6
]

# Create model
model = gp.Model("Hailong_Auto_Parts_Optimization")

# Decision Variables
# k[i, j]: Number of level i workers trained/assigned to line j
k = model.addVars(6, 3, vtype=GRB.INTEGER, name="k", lb=0)
# h[i, j]: Total weekly hours of level i workers on line j
h = model.addVars(6, 3, vtype=GRB.CONTINUOUS, name="h", lb=0)

# Set up the Objective Function
# Objective: Minimize Total Weekly Salary Expenditure + Training Costs
# Z = Sum_{i,j} (w_i * h_ij + c_ij * k_ij)
obj = gp.quicksum(W[i] * h[i, j] + C_cost[i][j] * k[i, j]
                  for i in range(6) for j in range(3))
model.setObjective(obj, GRB.MINIMIZE)

# Add Constraints

# 1. Worker count constraint: Total workers of level i assigned across all lines cannot exceed N_i
for i in range(6):
    model.addConstr(gp.quicksum(k[i, j] for j in range(3)) <= N[i], name=f"Total_Workers_L{i+1}")

# 2. Hours capacity link: Total hours on line j for level i cannot exceed capacity (40 hrs per worker)
for i in range(6):
    for j in range(3):
        model.addConstr(h[i, j] <= hours_per_week * k[i, j], name=f"Hours_Link_L{i+1}_Line{j}")

# 3. Demand satisfaction: Total production on each line must meet or exceed weekly demand
demands = [weekly_demand['A'], weekly_demand['B'], weekly_demand['C']]
for j in range(3):
    model.addConstr(gp.quicksum(R_prod[i][j] * h[i, j] for i in range(6)) >= demands[j], name=f"Demand_{part_types[j]}")

# Solve the model
model.optimize()

# Print results and output formatted FinalAnswer
if model.status == GRB.OPTIMAL:
    print(f"Minimal total expenditure: {model.objVal}")
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("Optimal solution not found.")