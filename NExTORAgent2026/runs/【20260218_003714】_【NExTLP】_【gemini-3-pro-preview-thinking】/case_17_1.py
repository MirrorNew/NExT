import gurobipy as gp
from gurobipy import GRB

# 1. Initialize Model
model = gp.Model("ColdChainLogistics")

# 2. Parameters
# Indices
tasks = ['A', 'B', 'C']  # i: 0, 1, 2
lines = ['A', 'B', 'C']  # j: 0, 1, 2 (Initial Cutting, Deep Freezing, Packaging)
days = [1, 2, 3]         # d: 1, 2, 3

num_tasks = len(tasks)
num_lines = len(lines)

# Processing times P[i][j]
# A: [3, 2, 1], B: [2, 4, 1], C: [4, 3, 2]
p = [
    [3, 2, 1], # A
    [2, 4, 1], # B
    [4, 3, 2]  # C
]

transfer_time = 0.5
M = 100 # Not strictly needed with indicator constraints, but good practice for bounds if needed
time_horizon = 72

# Shift definition
# Day 1: 0 - 14
# Day 2: 24 - 38
# Day 3: 48 - 62
def shift_start(d): return 24 * (d - 1)
def shift_end(d): return 24 * (d - 1) + 14

# 3. Variables
# s[i, j]: Start time of task i on line j
s = model.addVars(num_tasks, num_lines, vtype=GRB.CONTINUOUS, lb=0, ub=time_horizon, name="s")

# C_max: Makespan
C_max = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=time_horizon, name="C_max")

# x[i, k, j]: Binary, 1 if task i precedes task k on line j (for i < k)
x = {}
for j in range(num_lines):
    for i in range(num_tasks):
        for k in range(i + 1, num_tasks):
            x[i, k, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}_{j}")

# d_var[i, j, d]: Binary, 1 if task i on line j is assigned to day d
d_var = model.addVars(num_tasks, num_lines, days, vtype=GRB.BINARY, name="d_var")

# 4. Objective
model.setObjective(C_max, GRB.MINIMIZE)

# 5. Constraints

# 5.1 Precedence and Transfer Delay
# Line A -> Line B -> Line C
for i in range(num_tasks):
    # s_{i,1} >= s_{i,0} + p_{i,0} + 0.5
    model.addConstr(s[i, 1] >= s[i, 0] + p[i][0] + transfer_time, name=f"Precedence_A_B_Task_{tasks[i]}")
    # s_{i,2} >= s_{i,1} + p_{i,1} + 0.5
    model.addConstr(s[i, 2] >= s[i, 1] + p[i][1] + transfer_time, name=f"Precedence_B_C_Task_{tasks[i]}")

# 5.2 Machine Non-overlap (Disjunctive Constraints)
for j in range(num_lines):
    for i in range(num_tasks):
        for k in range(i + 1, num_tasks):
            # If x[i,k,j] == 1, then i precedes k: s[i,j] + p[i][j] <= s[k,j]
            model.addGenConstrIndicator(x[i, k, j], 1, s[i, j] + p[i][j] <= s[k, j])
            
            # If x[i,k,j] == 0, then k precedes i: s[k,j] + p[k][j] <= s[i, j]
            model.addGenConstrIndicator(x[i, k, j], 0, s[k, j] + p[k][j] <= s[i, j])

# 5.3 Daily Shift Windows & Assignment
for i in range(num_tasks):
    for j in range(num_lines):
        # Exactly one day assignment per task-line
        model.addConstr(gp.quicksum(d_var[i, j, d] for d in days) == 1, name=f"OneDay_Task{tasks[i]}_Line{lines[j]}")
        
        for d in days:
            start_window = shift_start(d)
            end_window = shift_end(d)
            
            # If assigned to day d, start time >= shift start
            model.addGenConstrIndicator(d_var[i, j, d], 1, s[i, j] >= start_window)
            
            # If assigned to day d, finish time <= shift end
            model.addGenConstrIndicator(d_var[i, j, d], 1, s[i, j] + p[i][j] <= end_window)

# 5.4 Makespan Definition
for i in range(num_tasks):
    model.addConstr(C_max >= s[i, 2] + p[i][2], name=f"Makespan_Task{tasks[i]}")

# 5.5 Time Horizon
model.addConstr(C_max <= time_horizon, name="MaxHorizon")

# 6. Solve
model.optimize()

# 7. Results
if model.status == GRB.OPTIMAL:
    print(f"Objective Value: {model.objVal}")
    for i in range(num_tasks):
        print(f"Task {tasks[i]}:")
        for j in range(num_lines):
            start = s[i, j].X
            finish = start + p[i][j]
            print(f"  Line {lines[j]}: {start:.2f} - {finish:.2f}")
    
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found.")