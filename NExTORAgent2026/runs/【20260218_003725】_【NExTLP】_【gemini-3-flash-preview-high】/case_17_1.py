import gurobipy as gp
from gurobipy import GRB
import itertools

# 1. Define all parameter matrices and data inputs.
# Data from problem description and Table 1
# Tasks: 0: A beef, 1: B cod, 2: C Shrimp
# Lines: 0: A (initial cutting), 1: B (deep freezing), 2: C (packaging)
tasks = [0, 1, 2]
task_names = ['A beef', 'B cod', 'C shrimp']
lines = [0, 1, 2]
line_names = ['Line A', 'Line B', 'Line C']
days = [0, 1, 2] # Days 1, 2, 3 (Time horizon up to 72 hours)

# p[task][line] is the processing time
p = [
    [3, 2, 1], # Beef
    [2, 4, 1], # Cod
    [4, 3, 2]  # Shrimp
]

# 2. Create Gurobi model.
model = gp.Model("ColdChainLogistics")

# 3. Create decision variables.
# s[i,j]: start time of task i on machine (line) j (absolute time from 0)
s = model.addVars(tasks, lines, lb=0, ub=72, name="s")
# d[i,j,d]: binary variable if task i on line j is processed on day d
d_var = model.addVars(tasks, lines, days, vtype=GRB.BINARY, name="d")
# x[i,k,j]: binary variable if task i precedes task k on line j
x = model.addVars(tasks, tasks, lines, vtype=GRB.BINARY, name="x")
# C_max: makespan (the last completion time on Line C)
C_max = model.addVar(lb=0, ub=72, name="C_max")

# 4. Set up the objective function.
model.setObjective(C_max, GRB.MINIMIZE)

# 5. Add constraints.

# Day Assignment and Shift Window
for i in tasks:
    for j in lines:
        # Each task must be assigned to exactly one day on each machine
        model.addConstr(gp.quicksum(d_var[i, j, day] for day in days) == 1)
        
        for day in days:
            # Shift window: 6:00 to 20:00 every day
            start_window = 24 * day + 6
            end_window = 24 * day + 20
            # Indicator constraints for shift windows
            model.addGenConstrIndicator(d_var[i, j, day], 1, s[i, j] >= start_window)
            model.addGenConstrIndicator(d_var[i, j, day], 1, s[i, j] + p[i][j] <= end_window)

# Precedence, Transfer Delay, and Sequential Order
for i in tasks:
    for j in range(2): # Lines 0->1 and 1->2
        day_j = gp.quicksum(day * d_var[i, j, day] for day in days)
        day_j_plus_1 = gp.quicksum(day * d_var[i, j+1, day] for day in days)
        # s[i,j+1] must start at least 0.5h (shift time) after task i finishes on line j.
        # This formula handles the 10-hour non-working gap (20:00 to 06:00) per day difference.
        model.addConstr(s[i, j+1] >= s[i, j] + p[i][j] + 0.5 + 10 * (day_j_plus_1 - day_j))

# Machine Non-overlap: each assembly line processes one batch at a time
M = 100 # Large constant for linearization
for j in lines:
    for i in tasks:
        for k in tasks:
            if i < k:
                # If x[i,k,j]=1, task i finishes before task k starts on machine j
                model.addConstr(s[i, j] + p[i][j] <= s[k, j] + M * (1 - x[i, k, j]))
                # If x[i,k,j]=0, task k finishes before task i starts on machine j
                model.addConstr(s[k, j] + p[k][j] <= s[i, j] + M * x[i, k, j])

# Global Capacity Limit: Max 2 tasks processed simultaneously across all lines
# Since we have 3 lines and 3 tasks, this constraint means at most 2 lines are active at once.
for i, k, m in itertools.permutations(tasks):
    # triplet (i, k, m) represents task i on line 0, task k on line 1, and task m on line 2
    # Create variables for finish times
    f_i0 = model.addVar(lb=0, ub=72, name=f"f_{i}_0")
    f_k1 = model.addVar(lb=0, ub=72, name=f"f_{k}_1")
    f_m2 = model.addVar(lb=0, ub=72, name=f"f_{m}_2")
    model.addConstr(f_i0 == s[i, 0] + p[i][0])
    model.addConstr(f_k1 == s[k, 1] + p[k][1])
    model.addConstr(f_m2 == s[m, 2] + p[m][2])
    
    # Calculate max start and min finish of the three processing intervals
    s_max = model.addVar(lb=0, ub=72, name=f"s_max_{i}_{k}_{m}")
    f_min = model.addVar(lb=0, ub=72, name=f"f_min_{i}_{k}_{m}")
    model.addGenConstrMax(s_max, [s[i, 0], s[k, 1], s[m, 2]])
    model.addGenConstrMin(f_min, [f_i0, f_k1, f_m2])
    
    # The condition that three intervals do not have a common intersection:
    # the maximum of the start times must be greater than or equal to the minimum of the finish times.
    model.addConstr(s_max >= f_min)

# Makespan Definition: total construction period is until line C finishes for all batches
for i in tasks:
    model.addConstr(C_max >= s[i, 2] + p[i][2])

# 6. Solve the model and print results.
model.setParam('OutputFlag', 0)
model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"FinalAnswer=【{C_max.X}】")