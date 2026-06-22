import gurobipy as gp
from gurobipy import GRB

# 1. Define Parameters List
Parameters_List = [
    {'Name': 'HeadNurseExperienceYears', 'Type': 'integer', 'Value': 15},
    {'Name': 'ArrivalHour', 'Type': 'integer', 'Value': 5},
    {'Name': 'NumShifts', 'Type': 'integer', 'Value': 5},
    {'Name': 'ShiftLengthHours', 'Type': 'integer', 'Value': 8},
    {'Name': 'ShiftTimes', 'Type': 'list', 'Value': ['2:00-10:00', '6:00-14:00', '10:00-18:00', '14:00-22:00', '18:00-2:00 (next day)']},
    {'Name': 'ShiftsPerWeek', 'Type': 'integer', 'Value': 5},
    {'Name': 'ConsecutiveWorkDays', 'Type': 'integer', 'Value': 5},
    {'Name': 'RestDays', 'Type': 'integer', 'Value': 2},
    {'Name': 'NursesRequired', 'Type': 'tuple', 'Value': {'6:00-10:00': 18, '10:00-14:00': 20, '14:00-18:00': 19, '18:00-22:00': 17, '22:00-6:00 (next day)': 12}}
]

# Extract specific parameters needed for the model
nurses_required = next(p['Value'] for p in Parameters_List if p['Name'] == 'NursesRequired')

# 2. Create Model
model = gp.Model("NurseScheduling")

# 3. Decision Variables
# x[d]: number of nurses whose first working day of the week is day d (0=Monday, ..., 6=Sunday)
x = model.addVars(7, vtype=GRB.INTEGER, lb=0, name="x")

# 4. Objective Function
# Minimize the total number of nurses
model.setObjective(gp.quicksum(x[d] for d in range(7)), GRB.MINIMIZE)

# 5. Constraints
# We iterate through each day t (0 to 6) and apply the coverage constraints provided in the context.
# Note: Indices are handled modulo 7.

for t in range(7):
    # Constraint 1: Coverage 6:00-10:00
    # Formula: x_t + x_{t-1} >= 18
    model.addConstr(x[t] + x[(t - 1) % 7] >= nurses_required['6:00-10:00'], name=f"Cov_6_10_day_{t}")

    # Constraint 2: Coverage 10:00-14:00
    # Formula: x_{t-1} + x_{t-2} >= 20
    model.addConstr(x[(t - 1) % 7] + x[(t - 2) % 7] >= nurses_required['10:00-14:00'], name=f"Cov_10_14_day_{t}")

    # Constraint 3: Coverage 14:00-18:00
    # Formula: x_{t-2} + x_{t-3} >= 19
    model.addConstr(x[(t - 2) % 7] + x[(t - 3) % 7] >= nurses_required['14:00-18:00'], name=f"Cov_14_18_day_{t}")

    # Constraint 4: Coverage 18:00-22:00
    # Formula: x_{t-3} + x_{t-4} >= 17
    model.addConstr(x[(t - 3) % 7] + x[(t - 4) % 7] >= nurses_required['18:00-22:00'], name=f"Cov_18_22_day_{t}")

    # Constraint 5: Coverage 22:00-6:00 (next day)
    # Formula: x_{t-4} + x_{t+1} >= 12
    model.addConstr(x[(t - 4) % 7] + x[(t + 1) % 7] >= nurses_required['22:00-6:00 (next day)'], name=f"Cov_22_06_day_{t}")

# 6. Solve the model
model.optimize()

# 7. Output Result
if model.status == GRB.OPTIMAL:
    total_nurses = int(model.objVal)
    print(f"Optimal Solution Found: Total Nurses = {total_nurses}")
    for d in range(7):
        print(f"Day {d} (start): {x[d].x}")
    print(f"FinalAnswer=【{total_nurses}】")
else:
    print("No optimal solution found.")