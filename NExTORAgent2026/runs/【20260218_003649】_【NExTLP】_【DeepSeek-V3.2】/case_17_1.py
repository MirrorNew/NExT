import gurobipy as gp
from gurobipy import GRB

# Parameters from the Parameters List
tasks = ['A', 'B', 'C']  # Task IDs
machines = [1, 2, 3]  # Machine IDs: 1=Line A, 2=Line B, 3=Line C
days = [1, 2, 3]  # Day indices (1, 2, 3)

# Processing times p[i][j] where i is task index, j is machine index (0-based)
# From Table_1_ProcessingTimes: [['A beef', 3, 2, 1], ['B cod', 2, 4, 1], ['C Shrimp', 4, 3, 2]]
p = {
    'A': {1: 3, 2: 2, 3: 1},
    'B': {1: 2, 2: 4, 3: 1},
    'C': {1: 4, 2: 3, 3: 2}
}

transfer_time = 0.5  # hours
shift_start_hour = 6  # 6:00
shift_end_hour = 20   # 20:00
daily_operation_hours = 14  # hours per day
max_days = 3
time_horizon = 72  # hours (3 days * 24 hours)

# Big M constant
M = time_horizon

# Create model
model = gp.Model("ColdChainScheduling")

# Decision variables
# Start times s[i][j]
s = {}
for i in tasks:
    for j in machines:
        s[i, j] = model.addVar(lb=0, ub=time_horizon, name=f"s_{i}_{j}")

# Makespan
C_max = model.addVar(lb=0, ub=time_horizon, name="C_max")

# Precedence variables x[i,k,j] for i != k on machine j
x = {}
for j in machines:
    for i in tasks:
        for k in tasks:
            if i != k:
                x[i, k, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}_{j}")

# Day assignment variables d[i,j,d]
d = {}
for i in tasks:
    for j in machines:
        for day in days:
            d[i, j, day] = model.addVar(vtype=GRB.BINARY, name=f"d_{i}_{j}_{day}")

model.update()

# Objective: minimize makespan
model.setObjective(C_max, GRB.MINIMIZE)

# Constraints

# 1. Precedence between machines for same task
for i in tasks:
    # Machine 2 must start after machine 1 finishes
    model.addConstr(s[i, 2] >= s[i, 1] + p[i][1], name=f"prec_{i}_1_2")
    # Machine 3 must start after machine 2 finishes
    model.addConstr(s[i, 3] >= s[i, 2] + p[i][2], name=f"prec_{i}_2_3")

# 2. Transfer delay (0.5h wait after each process)
for i in tasks:
    model.addConstr(s[i, 2] >= s[i, 1] + p[i][1] + transfer_time, name=f"transfer_{i}_1_2")
    model.addConstr(s[i, 3] >= s[i, 2] + p[i][2] + transfer_time, name=f"transfer_{i}_2_3")

# 3. Non-overlap on each machine (disjunctive constraints)
for j in machines:
    for i in tasks:
        for k in tasks:
            if i < k:  # Avoid duplicate constraints
                # If x[i,k,j] = 1, then i precedes k on machine j
                model.addConstr(s[i, j] + p[i][j] <= s[k, j] + M * (1 - x[i, k, j]), 
                               name=f"no_overlap_{i}_{k}_{j}_1")
                # If x[i,k,j] = 0, then k precedes i on machine j
                model.addConstr(s[k, j] + p[k][j] <= s[i, j] + M * x[i, k, j], 
                               name=f"no_overlap_{i}_{k}_{j}_2")

# 4. Daily shift windows using indicator constraints
for i in tasks:
    for j in machines:
        # Each task on each machine assigned to exactly one day
        model.addConstr(gp.quicksum(d[i, j, day] for day in days) == 1, 
                       name=f"one_day_{i}_{j}")
        
        for day in days:
            day_start = 24 * (day - 1)  # Start of day in hours from time 0
            shift_start = day_start + shift_start_hour  # 6:00 on that day
            shift_end = day_start + shift_end_hour  # 20:00 on that day
            
            # If d[i,j,day] = 1, then s[i,j] >= shift_start
            model.addGenConstrIndicator(d[i, j, day], 1, s[i, j] >= shift_start,
                                       name=f"shift_start_{i}_{j}_{day}")
            
            # If d[i,j,day] = 1, then s[i,j] + p[i][j] <= shift_end
            model.addGenConstrIndicator(d[i, j, day], 1, s[i, j] + p[i][j] <= shift_end,
                                       name=f"shift_end_{i}_{j}_{day}")

# 5. Makespan definition
for i in tasks:
    model.addConstr(C_max >= s[i, 3] + p[i][3], name=f"makespan_{i}")

# 6. Time horizon (already enforced by variable bounds)
# C_max <= 72 is already in variable definition
# s[i,j] <= 72 - p[i][j] is not needed due to shift constraints

# Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"Minimum makespan (C_max): {C_max.X:.2f} hours")
    
    print("\nSchedule:")
    for i in tasks:
        for j in machines:
            start = s[i, j].X
            finish = start + p[i][j]
            # Determine which day
            day_assigned = None
            for day in days:
                if d[i, j, day].X > 0.5:
                    day_assigned = day
                    break
            
            machine_name = f"Line {['A', 'B', 'C'][j-1]}"
            task_name = f"{i} ({['beef', 'cod', 'shrimp'][tasks.index(i)]})"
            process_name = ["initial cutting", "deep freezing", "packaging"][j-1]
            
            print(f"Task {task_name} on {machine_name} ({process_name}): "
                  f"Day {day_assigned}, Start={start:.2f}h, Finish={finish:.2f}h")
    
    # Calculate total time from start of first task to end of last task
    first_start = min(s[i, j].X for i in tasks for j in machines)
    last_finish = max(s[i, 3].X + p[i][3] for i in tasks)
    
    print(f"\nFirst start time: {first_start:.2f}h")
    print(f"Last finish time: {last_finish:.2f}h")
    print(f"Total construction period: {C_max.X:.2f} hours")
    
    # Output the final answer as required
    print(f"FinalAnswer=【{C_max.X:.2f}】")
    
else:
    print("No optimal solution found")
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible")
        model.computeIIS()
        model.write("model.ilp")
    print(f"FinalAnswer=【-1】")