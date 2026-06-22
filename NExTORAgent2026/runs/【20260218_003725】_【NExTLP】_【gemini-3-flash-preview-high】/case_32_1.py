import gurobipy as gp
from gurobipy import GRB

# Create a new Gurobi model
model = gp.Model("DongfengExpressSorting")

# Parameters from the provided context
num_sorting_machines = 11
sorter_capacity_per_hour = 500
full_time_daily_pay = 150
part_time_daily_pay = 80

# Arrivals from Table C-33
# Arrival units occurring before specific time slots
# Before 10:00: 5000
# 10:00-11:00: 4000
# 11:00-12:00: 3000
# 12:00-13:00: 4000
# 13:00-14:00: 2500
# 14:00-15:00: 3000
# 15:00-16:00: 4000
# 16:00-17:00: 4500
# 17:00-18:00: 3500
# 18:00-19:00: 2500
# Total Arrivals: 36,000
arrivals = [5000, 4000, 3000, 4000, 2500, 3000, 4000, 4500, 3500, 2500]

# Cumulative arrivals A[t] represent the max quantity available to be processed by end of hour t
# Slot 0 (10-11) can process what arrived before 10:00
# Slot 1 (11-12) can process what arrived before 11:00
A = []
cumulative = 0
for arr in arrivals:
    cumulative += arr
    A.append(cumulative)

# Decision Variables
# x1: full-time (10:00-18:00), x2: full-time (11:00-19:00), x3: full-time (12:00-20:00)
x = model.addVars(3, vtype=GRB.INTEGER, name="x", lb=0)
# y1: part-time (13:00-18:00), y2: part-time (14:00-19:00), y3: part-time (15:00-20:00)
y = model.addVars(3, vtype=GRB.INTEGER, name="y", lb=0)

# S[t]: Number of express parcels processed in time slot t (0 for 10-11, ..., 9 for 19-20)
S = model.addVars(10, vtype=GRB.INTEGER, name="S", lb=0)

# Objective: Minimize total daily wage cost
model.setObjective(full_time_daily_pay * (x[0] + x[1] + x[2]) + 
                   part_time_daily_pay * (y[0] + y[1] + y[2]), GRB.MINIMIZE)

# Define workforce at each hour based on shifts
# Periods: 10-11(0), 11-12(1), 12-13(2), 13-14(3), 14-15(4), 15-16(5), 16-17(6), 17-18(7), 18-19(8), 19-20(9)
W = [0] * 10
W[0] = x[0]
W[1] = x[0] + x[1]
W[2] = x[0] + x[1] + x[2]
W[3] = x[0] + x[1] + x[2] + y[0]
W[4] = x[0] + x[1] + x[2] + y[0] + y[1]
W[5] = x[0] + x[1] + x[2] + y[0] + y[1] + y[2]
W[6] = x[0] + x[1] + x[2] + y[0] + y[1] + y[2]
W[7] = x[0] + x[1] + x[2] + y[0] + y[1] + y[2]
W[8] = x[1] + x[2] + y[1] + y[2]
W[9] = x[2] + y[2]

# Constraints
for t in range(10):
    # Machine availability (Each employee uses one machine, max 11 machines)
    model.addConstr(W[t] <= num_sorting_machines, name=f"MachineLimit_t{t}")
    
    # Processing Capacity (S[t] <= 500 * Number of Employees)
    model.addConstr(S[t] <= sorter_capacity_per_hour * W[t], name=f"CapacityLimit_t{t}")
    
    # Cumulative processing cannot exceed cumulative arrivals
    # S[0] <= A[0], S[0]+S[1] <= A[1], etc.
    model.addConstr(gp.quicksum(S[i] for i in range(t + 1)) <= A[t], name=f"ArrivalConstraint_t{t}")

# Deadlines
# 1. All express mails arriving before 12:00 must be processed before 14:00 (end of slot 3)
# Arrivals before 12:00 = A[2] = 12,000
model.addConstr(gp.quicksum(S[i] for i in range(4)) >= 12000, name="Deadline_1400")

# 2. Those arriving before 15:00 must be processed before 17:00 (end of slot 6)
# Arrivals before 15:00 = A[5] = 21,500
model.addConstr(gp.quicksum(S[i] for i in range(7)) >= 21500, name="Deadline_1700")

# 3. All express mails arriving on the same day must be processed before 20:00 (end of slot 9)
# Total arrivals = 36,000
model.addConstr(gp.quicksum(S[i] for i in range(10)) == 36000, name="FinalDeadline_2000")

# Solve the model
model.optimize()

# Output Results
if model.status == GRB.OPTIMAL:
    print(f"Optimal total daily expenditure: {model.ObjVal}")
    print(f"Full-time employees (x1, x2, x3): {x[0].X}, {x[1].X}, {x[2].X}")
    print(f"Part-time employees (y1, y2, y3): {y[0].X}, {y[1].X}, {y[2].X}")
    print(f"FinalAnswer=【{int(model.ObjVal)}】")
else:
    print("No optimal solution found.")