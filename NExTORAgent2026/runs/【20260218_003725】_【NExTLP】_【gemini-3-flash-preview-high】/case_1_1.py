import gurobipy as gp
from gurobipy import GRB

# Create a new Gurobi model
model = gp.Model("NurseScheduling")

# Parameter definitions based on the problem
# Shifts are 8 hours long, 5 shifts per day.
# Nurses work 5 consecutive days, each starting on a specific day of the week.
# Shift 1: 2:00-10:00, Shift 2: 6:00-14:00, Shift 3: 10:00-18:00, Shift 4: 14:00-22:00, Shift 5: 18:00-2:00 (next day)
# Requirement for each time period:
requirements = {
    '6:00-10:00': 18,
    '10:00-14:00': 20,
    '14:00-18:00': 19,
    '18:00-22:00': 17,
    '22:00-6:00': 12
}

# Decision Variables
# x[i] is the number of nurses whose first working day of the week is day i (0=Monday, ..., 6=Sunday)
x = model.addVars(7, vtype=GRB.INTEGER, lb=0, name="x")

# Objective function: Minimize the total number of nurses
model.setObjective(gp.quicksum(x[i] for i in range(7)), GRB.MINIMIZE)

# Constraints
# For each day t (0 to 6), the time period requirements must be met.
# According to the scheduling plan:
# Nurses on Shift 1 on day t started on day t (index i=t)
# Nurses on Shift 2 on day t started on day t-1 (index i=(t-1)%7)
# Nurses on Shift 3 on day t started on day t-2 (index i=(t-2)%7)
# Nurses on Shift 4 on day t started on day t-3 (index i=(t-3)%7)
# Nurses on Shift 5 on day t started on day t-4 (index i=(t-4)%7)

for i in range(7):
    # Coverage 6:00–10:00 (Day t): Shift 1 and Shift 2 are on duty
    model.addConstr(x[i] + x[(i-1)%7] >= requirements['6:00-10:00'], name=f"cov_6_10_{i}")
    
    # Coverage 10:00–14:00 (Day t): Shift 2 and Shift 3 are on duty
    model.addConstr(x[(i-1)%7] + x[(i-2)%7] >= requirements['10:00-14:00'], name=f"cov_10_14_{i}")
    
    # Coverage 14:00–18:00 (Day t): Shift 3 and Shift 4 are on duty
    model.addConstr(x[(i-2)%7] + x[(i-3)%7] >= requirements['14:00-18:00'], name=f"cov_14_18_{i}")
    
    # Coverage 18:00–22:00 (Day t): Shift 4 and Shift 5 are on duty
    model.addConstr(x[(i-3)%7] + x[(i-4)%7] >= requirements['18:00-22:00'], name=f"cov_18_22_{i}")
    
    # Coverage 22:00–06:00 (Day t to next day): 
    # This period is covered by Shift 5 (Day t) and Shift 1 (Day t+1).
    # Shift 5 (Day t) started on Day t-4: x[(i-4)%7]
    # Shift 1 (Day t+1) started on Day t+1: x[(i+1)%7]
    model.addConstr(x[(i-4)%7] + x[(i+1)%7] >= requirements['22:00-6:00'], name=f"cov_22_6_{i}")

# Optimize the model
model.optimize()

# Print the final answer
if model.status == GRB.OPTIMAL:
    final_answer = int(model.objVal)
    print(f"FinalAnswer=【{final_answer}】")