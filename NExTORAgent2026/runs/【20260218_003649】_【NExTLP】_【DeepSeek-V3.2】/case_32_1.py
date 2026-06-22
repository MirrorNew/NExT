import gurobipy as gp

# Define parameters from the Parameters List
num_sorting_machines = 11
sorter_capacity_per_hour = 500
full_time_daily_pay = 150
part_time_daily_pay = 80

# Arrivals per time period (periods p=1..10 represent 10:00-11:00, ..., 19:00-20:00)
arrivals = [
    5000,  # before 10:00, can be processed from period 1
    4000,  # 10:00-11:00
    3000,  # 11:00-12:00
    4000,  # 12:00-13:00
    2500,  # 13:00-14:00
    3000,  # 14:00-15:00
    4000,  # 15:00-16:00
    4500,  # 16:00-17:00
    3500,  # 17:00-18:00
    2500   # 18:00-19:00
]

# Total parcels for the day
total_parcels = sum(arrivals)

# Indicator matrices for each shift (10 periods)
I1 = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  # x1: 10:00-18:00
I2 = [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  # x2: 11:00-19:00
I3 = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  # x3: 12:00-20:00

J1 = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  # y1: 13:00-18:00
J2 = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  # y2: 14:00-19:00
J3 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # y3: 15:00-20:00

# Create model
model = gp.Model("Dongfeng_Express_Staffing")

# Decision variables
x1 = model.addVar(vtype=gp.GRB.INTEGER, name="x1")
x2 = model.addVar(vtype=gp.GRB.INTEGER, name="x2")
x3 = model.addVar(vtype=gp.GRB.INTEGER, name="x3")
y1 = model.addVar(vtype=gp.GRB.INTEGER, name="y1")
y2 = model.addVar(vtype=gp.GRB.INTEGER, name="y2")
y3 = model.addVar(vtype=gp.GRB.INTEGER, name="y3")

# Objective: minimize total daily wage cost
model.setObjective(
    full_time_daily_pay * (x1 + x2 + x3) + 
    part_time_daily_pay * (y1 + y2 + y3),
    gp.GRB.MINIMIZE
)

# Machine availability constraints for each time period
for p in range(10):
    total_working = (
        x1 * I1[p] + x2 * I2[p] + x3 * I3[p] +
        y1 * J1[p] + y2 * J2[p] + y3 * J3[p]
    )
    model.addConstr(total_working <= num_sorting_machines, 
                    name=f"Machine_availability_period_{p+1}")

# Deadline constraints
# 1. Parcels arriving before 12:00 must be processed by 14:00
# Cumulative arrivals up to period 3 (before 12:00): 5000+4000+3000 = 12000
# Must be processed by end of period 4 (14:00)
sum_E_first_4 = gp.quicksum(
    x1 * I1[p] + x2 * I2[p] + x3 * I3[p] +
    y1 * J1[p] + y2 * J2[p] + y3 * J3[p]
    for p in range(4)  # periods 1-4
)
model.addConstr(sorter_capacity_per_hour * sum_E_first_4 >= 12000, 
                name="Deadline_before_14:00")

# 2. Parcels arriving before 15:00 must be processed by 17:00
# Cumulative arrivals up to period 6 (before 15:00): 
# 12000 + 4000 + 2500 + 3000 = 21500
# Must be processed by end of period 7 (17:00)
sum_E_first_7 = gp.quicksum(
    x1 * I1[p] + x2 * I2[p] + x3 * I3[p] +
    y1 * J1[p] + y2 * J2[p] + y3 * J3[p]
    for p in range(7)  # periods 1-7
)
model.addConstr(sorter_capacity_per_hour * sum_E_first_7 >= 21500, 
                name="Deadline_before_17:00")

# 3. All parcels must be processed by 20:00
# Total parcels: 36000
sum_E_all = gp.quicksum(
    x1 * I1[p] + x2 * I2[p] + x3 * I3[p] +
    y1 * J1[p] + y2 * J2[p] + y3 * J3[p]
    for p in range(10)  # all 10 periods
)
model.addConstr(sorter_capacity_per_hour * sum_E_all >= total_parcels, 
                name="Deadline_before_20:00")

# Solve the model
model.optimize()

# Print results
if model.status == gp.GRB.OPTIMAL:
    print("Optimal solution found:")
    print(f"x1 (full-time 10:00-18:00) = {x1.x}")
    print(f"x2 (full-time 11:00-19:00) = {x2.x}")
    print(f"x3 (full-time 12:00-20:00) = {x3.x}")
    print(f"y1 (part-time 13:00-18:00) = {y1.x}")
    print(f"y2 (part-time 14:00-19:00) = {y2.x}")
    print(f"y3 (part-time 15:00-20:00) = {y3.x}")
    print(f"Total full-time employees: {x1.x + x2.x + x3.x}")
    print(f"Total part-time employees: {y1.x + y2.x + y3.x}")
    print(f"Total daily cost: {model.objVal}")
    
    # Format answer as required
    full_time_total = int(x1.x + x2.x + x3.x)
    part_time_total = int(y1.x + y2.x + y3.x)
    print(f"FinalAnswer=【full-time: {full_time_total}, part-time: {part_time_total}, total cost: {model.objVal}】")
else:
    print("No optimal solution found")
    print(f"FinalAnswer=【No feasible solution】")