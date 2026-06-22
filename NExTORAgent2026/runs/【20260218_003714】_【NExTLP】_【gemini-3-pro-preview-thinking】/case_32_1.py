import gurobipy as gp
from gurobipy import GRB

# Create the model
model = gp.Model("DongfengExpressSorting")

# Parameters
# Daily pay
pay_ft = 150
pay_pt = 80

# Sorting capacity per machine per hour
capacity = 500

# Number of sorting machines
num_machines = 11

# Demand Data (Number of arriving express parcels)
# Time periods map to index representing the start hour: e.g., 10 -> 10:00-11:00
# Arrive before 10:00: 5000
demand_data = {
    'before_10': 5000,
    10: 4000,
    11: 3000,
    12: 4000,
    13: 2500,
    14: 3000,
    15: 4000,
    16: 4500,
    17: 3500,
    18: 2500
}

# Decision Variables
# Full-time employees
# x1: 10:00-18:00
# x2: 11:00-19:00
# x3: 12:00-20:00
x1 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x1")
x2 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x2")
x3 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x3")

# Part-time employees
# y1: 13:00-18:00 (5 hours)
# y2: 14:00-19:00 (5 hours)
# y3: 15:00-20:00 (5 hours)
y1 = model.addVar(vtype=GRB.INTEGER, lb=0, name="y1")
y2 = model.addVar(vtype=GRB.INTEGER, lb=0, name="y2")
y3 = model.addVar(vtype=GRB.INTEGER, lb=0, name="y3")

# Objective Function: Minimize total daily wage cost
model.setObjective(pay_ft * (x1 + x2 + x3) + pay_pt * (y1 + y2 + y3), GRB.MINIMIZE)

# Constraints

# 1. Machine Availability Constraints
# The number of people working in any specific hour interval [t, t+1] must not exceed 11.
# Full-time work 8 hours, Part-time work 5 hours.

# 10:00 - 11:00: x1
model.addConstr(x1 <= num_machines, "Machine_10_11")

# 11:00 - 12:00: x1, x2
model.addConstr(x1 + x2 <= num_machines, "Machine_11_12")

# 12:00 - 13:00: x1, x2, x3
model.addConstr(x1 + x2 + x3 <= num_machines, "Machine_12_13")

# 13:00 - 14:00: x1, x2, x3, y1
model.addConstr(x1 + x2 + x3 + y1 <= num_machines, "Machine_13_14")

# 14:00 - 15:00: x1, x2, x3, y1, y2
model.addConstr(x1 + x2 + x3 + y1 + y2 <= num_machines, "Machine_14_15")

# 15:00 - 16:00: x1, x2, x3, y1, y2, y3
model.addConstr(x1 + x2 + x3 + y1 + y2 + y3 <= num_machines, "Machine_15_16")

# 16:00 - 17:00: x1, x2, x3, y1, y2, y3
model.addConstr(x1 + x2 + x3 + y1 + y2 + y3 <= num_machines, "Machine_16_17")

# 17:00 - 18:00: x1, x2, x3, y1, y2, y3
model.addConstr(x1 + x2 + x3 + y1 + y2 + y3 <= num_machines, "Machine_17_18")

# 18:00 - 19:00: x2, x3, y2, y3 (x1 and y1 end at 18:00)
model.addConstr(x2 + x3 + y2 + y3 <= num_machines, "Machine_18_19")

# 19:00 - 20:00: x3, y3 (x2 and y2 end at 19:00)
model.addConstr(x3 + y3 <= num_machines, "Machine_19_20")

# 2. Processing Deadline Constraints
# Capacity: 500 parcels per person-hour

# Deadline 14:00
# All express mails arriving before 12:00 must be processed before 14:00.
# Arriving before 12:00 = 5000 (before 10) + 4000 (10-11) + 3000 (11-12) = 12000
# Processing window: 10:00 to 14:00 (4 hours total)
# Hours worked by each shift type in [10:00, 14:00]:
# x1 (10-18): 4 hours
# x2 (11-19): 3 hours (starts 11)
# x3 (12-20): 2 hours (starts 12)
# y1 (13-18): 1 hour (starts 13)
# y2 (14-19): 0 hours
# y3 (15-20): 0 hours
demand_upto_12 = demand_data['before_10'] + demand_data[10] + demand_data[11]
model.addConstr(capacity * (4*x1 + 3*x2 + 2*x3 + 1*y1) >= demand_upto_12, "Deadline_14_Constraint")

# Deadline 17:00
# All express mails arriving before 15:00 must be processed before 17:00.
# Arriving before 15:00 = 12000 (prev) + 4000 (12-13) + 2500 (13-14) + 3000 (14-15) = 21500
# Processing window: 10:00 to 17:00 (7 hours total)
# Hours worked by each shift type in [10:00, 17:00]:
# x1: 7 hours
# x2: 6 hours
# x3: 5 hours
# y1: 4 hours
# y2: 3 hours (starts 14)
# y3: 2 hours (starts 15)
demand_upto_15 = demand_upto_12 + demand_data[12] + demand_data[13] + demand_data[14]
model.addConstr(capacity * (7*x1 + 6*x2 + 5*x3 + 4*y1 + 3*y2 + 2*y3) >= demand_upto_15, "Deadline_17_Constraint")

# Deadline 20:00 (End of day)
# All express mails arriving on the same day must be processed before 20:00.
# Total Demand = 21500 (prev) + 4000 (15-16) + 4500 (16-17) + 3500 (17-18) + 2500 (18-19) = 36000
# Processing window: 10:00 to 20:00 (10 hours total)
# Hours worked by each shift type in [10:00, 20:00]:
# x1: 8 hours (ends 18)
# x2: 8 hours (ends 19)
# x3: 8 hours (ends 20)
# y1: 5 hours (ends 18)
# y2: 5 hours (ends 19)
# y3: 5 hours (ends 20)
total_demand = demand_upto_15 + demand_data[15] + demand_data[16] + demand_data[17] + demand_data[18]
model.addConstr(capacity * (8*x1 + 8*x2 + 8*x3 + 5*y1 + 5*y2 + 5*y3) >= total_demand, "Deadline_20_Constraint")

# Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print("\nOptimal Solution Found:")
    print(f"Full-time 10-18 (x1): {x1.x}")
    print(f"Full-time 11-19 (x2): {x2.x}")
    print(f"Full-time 12-20 (x3): {x3.x}")
    print(f"Part-time 13-18 (y1): {y1.x}")
    print(f"Part-time 14-19 (y2): {y2.x}")
    print(f"Part-time 15-20 (y3): {y3.x}")
    print(f"Total Cost: {model.objVal}")
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found.")