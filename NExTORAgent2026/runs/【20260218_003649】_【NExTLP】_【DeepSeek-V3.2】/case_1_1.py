import gurobipy as gp
from gurobipy import GRB

# Define parameters from the Parameters List
# The nurse requirements for each time period
requirements = {
    '6:00-10:00': 18,
    '10:00-14:00': 20,
    '14:00-18:00': 19,
    '18:00-22:00': 17,
    '22:00-6:00 (next day)': 12
}

# Create model
model = gp.Model("NurseScheduling")

# Decision variables: x_d = number of nurses starting on day d (d=1..7, Monday=1)
x = model.addVars(7, vtype=GRB.INTEGER, name="x")

# Objective: minimize total number of nurses
model.setObjective(gp.quicksum(x[d] for d in range(7)), GRB.MINIMIZE)

# Add coverage constraints for each day t (t=0..6 representing Monday..Sunday)
for t in range(7):
    # Day indices modulo 7 (0-based)
    # Constraint 1: 6:00-10:00 coverage: x_t + x_{t-1} >= 18
    model.addConstr(x[t] + x[(t-1) % 7] >= requirements['6:00-10:00'], 
                   f"Coverage_6_10_day_{t+1}")
    
    # Constraint 2: 10:00-14:00 coverage: x_{t-1} + x_{t-2} >= 20
    model.addConstr(x[(t-1) % 7] + x[(t-2) % 7] >= requirements['10:00-14:00'], 
                   f"Coverage_10_14_day_{t+1}")
    
    # Constraint 3: 14:00-18:00 coverage: x_{t-2} + x_{t-3} >= 19
    model.addConstr(x[(t-2) % 7] + x[(t-3) % 7] >= requirements['14:00-18:00'], 
                   f"Coverage_14_18_day_{t+1}")
    
    # Constraint 4: 18:00-22:00 coverage: x_{t-3} + x_{t-4} >= 17
    model.addConstr(x[(t-3) % 7] + x[(t-4) % 7] >= requirements['18:00-22:00'], 
                   f"Coverage_18_22_day_{t+1}")
    
    # Constraint 5: 22:00-6:00 (next day) coverage: x_{t-4} + x_{t+1} >= 12
    model.addConstr(x[(t-4) % 7] + x[(t+1) % 7] >= requirements['22:00-6:00 (next day)'], 
                   f"Coverage_22_6_day_{t+1}")

# Non-negativity constraints are already enforced by vtype=INTEGER with default lb=0

# Solve the model
model.optimize()

# Check if solution is found
if model.status == GRB.OPTIMAL:
    total_nurses = int(model.ObjVal)
    print("Optimal solution found!")
    print(f"Minimum number of nurses required: {total_nurses}")
    
    # Print detailed schedule
    print("\nNumber of nurses starting on each day:")
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for d in range(7):
        print(f"{days[d]}: {int(x[d].X)} nurses")
    
    # Verify coverage for each time period on Monday (as example)
    print("\nCoverage verification for Monday (day 1):")
    print(f"6:00-10:00: x1 + x7 = {int(x[0].X)} + {int(x[6].X)} = {int(x[0].X + x[6].X)} (required: {requirements['6:00-10:00']})")
    print(f"10:00-14:00: x7 + x6 = {int(x[6].X)} + {int(x[5].X)} = {int(x[6].X + x[5].X)} (required: {requirements['10:00-14:00']})")
    print(f"14:00-18:00: x6 + x5 = {int(x[5].X)} + {int(x[4].X)} = {int(x[5].X + x[4].X)} (required: {requirements['14:00-18:00']})")
    print(f"18:00-22:00: x5 + x4 = {int(x[4].X)} + {int(x[3].X)} = {int(x[4].X + x[3].X)} (required: {requirements['18:00-22:00']})")
    print(f"22:00-6:00: x4 + x2 = {int(x[3].X)} + {int(x[1].X)} = {int(x[3].X + x[1].X)} (required: {requirements['22:00-6:00 (next day)']})")
    
    print(f"\nFinalAnswer=【{total_nurses}】")
else:
    print("No optimal solution found. Status:", model.status)
    print(f"FinalAnswer=【No feasible solution】")