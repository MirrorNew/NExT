import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters (given)
# =========================
year_founded = 2010
max_daily_parcels = 2000000
num_sorters = 11
sorter_capacity_per_hour = 500
avg_daily_parcels = 35000
ecommerce_ratio = 0.78
num_sorting_machines = 11
full_time_shifts = [[10, 18], [11, 19], [12, 20]]
full_time_daily_pay = 150
part_time_shifts = [[13, 18], [14, 19], [15, 20]]
part_time_hours_per_day = 5
part_time_daily_pay = 80
processing_deadlines = [[12, 14], [15, 17], [24, 20]]
Table_1_C33 = {
    'Before 10:00': 5000,
    '10:00-11:00': 4000,
    '11:00-12:00': 3000,
    '12:00-13:00': 4000,
    '13:00-14:00': 2500,
    '14:00-15:00': 3000,
    '15:00-16:00': 4000,
    '16:00-17:00': 4500,
    '17:00-18:00': 3500,
    '18:00-19:00': 2500
}
Table_2_C12 = {
    'times': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'x1': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    'x2': [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    'x3': [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
}

# =========================
# 2. Time periods and availability matrices
# =========================
# Define processing periods p = 1..10 corresponding to:
# 1: 10-11, 2: 11-12, 3: 12-13, 4: 13-14, 5: 14-15,
# 6: 15-16, 7: 16-17, 8: 17-18, 9: 18-19, 10: 19-20
periods = list(range(1, 11))

# Build availability for full-time shifts from Table_2_C12
times = Table_2_C12['times']  # [10,...,20]
x1_avail_times = Table_2_C12['x1']
x2_avail_times = Table_2_C12['x2']
x3_avail_times = Table_2_C12['x3']

# Map each period p to starting time hour
period_start_time = {
    1: 10,
    2: 11,
    3: 12,
    4: 13,
    5: 14,
    6: 15,
    7: 16,
    8: 17,
    9: 18,
    10: 19
}

# Availability matrices I1, I2, I3 for full-time
I1 = {}
I2 = {}
I3 = {}
for p in periods:
    t = period_start_time[p]
    idx = times.index(t)
    I1[p] = x1_avail_times[idx]
    I2[p] = x2_avail_times[idx]
    I3[p] = x3_avail_times[idx]

# Availability matrices for part-time shifts from part_time_shifts
# part_time_shifts: [[13,18],[14,19],[15,20]]
J1 = {}
J2 = {}
J3 = {}
for p in periods:
    t = period_start_time[p]
    # y1: 13-18 -> active at t = 13,14,15,16,17
    J1[p] = 1 if (t >= part_time_shifts[0][0]) and (t < part_time_shifts[0][1]) else 0
    # y2: 14-19 -> active at t = 14,15,16,17,18
    J2[p] = 1 if (t >= part_time_shifts[1][0]) and (t < part_time_shifts[1][1]) else 0
    # y3: 15-20 -> active at t = 15,16,17,18,19
    J3[p] = 1 if (t >= part_time_shifts[2][0]) and (t < part_time_shifts[2][1]) else 0

# =========================
# 3. Gurobi model
# =========================
model = gp.Model("Express_Sorting_Staffing")

# Decision variables: non-negative integers
x1 = model.addVar(vtype=GRB.INTEGER, name="x1", lb=0)  # full-time 10-18
x2 = model.addVar(vtype=GRB.INTEGER, name="x2", lb=0)  # full-time 11-19
x3 = model.addVar(vtype=GRB.INTEGER, name="x3", lb=0)  # full-time 12-20
y1 = model.addVar(vtype=GRB.INTEGER, name="y1", lb=0)  # part-time 13-18
y2 = model.addVar(vtype=GRB.INTEGER, name="y2", lb=0)  # part-time 14-19
y3 = model.addVar(vtype=GRB.INTEGER, name="y3", lb=0)  # part-time 15-20

model.update()

# =========================
# 4. Objective function
# =========================
model.setObjective(
    full_time_daily_pay * (x1 + x2 + x3) +
    part_time_daily_pay * (y1 + y2 + y3),
    GRB.MINIMIZE
)

# =========================
# 5. Constraints
# =========================

# (1) Machine availability in each processing period
# x1*I1(p) + x2*I2(p) + x3*I3(p) + y1*J1(p) + y2*J2(p) + y3*J3(p) <= num_sorting_machines
for p in periods:
    model.addConstr(
        x1 * I1[p] + x2 * I2[p] + x3 * I3[p] +
        y1 * J1[p] + y2 * J2[p] + y3 * J3[p] <= num_sorting_machines,
        name=f"Machine_availability_p{p}"
    )

# Constants for deadlines from the context
demand_before_12 = Table_1_C33['Before 10:00'] + Table_1_C33['10:00-11:00'] + Table_1_C33['11:00-12:00']  # 12000
demand_before_15 = (
    Table_1_C33['Before 10:00'] +
    Table_1_C33['10:00-11:00'] +
    Table_1_C33['11:00-12:00'] +
    Table_1_C33['12:00-13:00'] +
    Table_1_C33['13:00-14:00'] +
    Table_1_C33['14:00-15:00']
)  # 21500
total_daily_demand = 36000

# (2) Deadline before 14:00 (periods 1..4)
expr_14 = gp.LinExpr()
for p in range(1, 5):  # 1..4
    expr_14 += sorter_capacity_per_hour * (
        x1 * I1[p] + x2 * I2[p] + x3 * I3[p] +
        y1 * J1[p] + y2 * J2[p] + y3 * J3[p]
    )
model.addConstr(
    expr_14 >= demand_before_12,
    name="Deadline_before_14"
)

# (3) Deadline before 17:00 (periods 1..7)
expr_17 = gp.LinExpr()
for p in range(1, 8):  # 1..7
    expr_17 += sorter_capacity_per_hour * (
        x1 * I1[p] + x2 * I2[p] + x3 * I3[p] +
        y1 * J1[p] + y2 * J2[p] + y3 * J3[p]
    )
model.addConstr(
    expr_17 >= demand_before_15,
    name="Deadline_before_17"
)

# (4) Deadline before 20:00 (periods 1..10)
expr_20 = gp.LinExpr()
for p in periods:  # 1..10
    expr_20 += sorter_capacity_per_hour * (
        x1 * I1[p] + x2 * I2[p] + x3 * I3[p] +
        y1 * J1[p] + y2 * J2[p] + y3 * J3[p]
    )
model.addConstr(
    expr_20 >= total_daily_demand,
    name="Deadline_before_20"
)

# =========================
# 6. Solve the model
# =========================
model.optimize()

# =========================
# 7. Print results
# =========================
if model.status == GRB.OPTIMAL:
    x1_val = int(round(x1.X))
    x2_val = int(round(x2.X))
    x3_val = int(round(x3.X))
    y1_val = int(round(y1.X))
    y2_val = int(round(y2.X))
    y3_val = int(round(y3.X))
    total_cost = full_time_daily_pay * (x1_val + x2_val + x3_val) + \
                 part_time_daily_pay * (y1_val + y2_val + y3_val)

    print("Optimal staffing plan:")
    print(f"  Full-time 10:00-18:00 (x1): {x1_val}")
    print(f"  Full-time 11:00-19:00 (x2): {x2_val}")
    print(f"  Full-time 12:00-20:00 (x3): {x3_val}")
    print(f"  Part-time 13:00-18:00 (y1): {y1_val}")
    print(f"  Part-time 14:00-19:00 (y2): {y2_val}")
    print(f"  Part-time 15:00-20:00 (y3): {y3_val}")
    print(f"Minimum total daily wage cost: {total_cost}")

    # FinalAnswer: the minimum total daily wage cost
    print(f"FinalAnswer=【{total_cost}】")
else:
    # If not optimal, still output something explicit
    print("No optimal solution found.")
    print("FinalAnswer=【None】")