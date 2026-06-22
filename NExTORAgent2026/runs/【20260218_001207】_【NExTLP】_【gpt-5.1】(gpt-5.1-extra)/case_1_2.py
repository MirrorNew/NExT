import gurobipy as gp

# Nurse scheduling MILP using the validated mathematical model
# Minimize the total number of nurses while satisfying demand in each time period
# over a 7-day repeating cycle, with each nurse working 5 consecutive days then 2 days off.

# ----------------------------------------------------------------------
# 1. Define all parameter matrices and data inputs (from Parameters List)
# ----------------------------------------------------------------------
head_nurse_tenure_years = 15
head_nurse_arrival_time = ['5:00']

time_period_labels = [
    '6:00-10:00',
    '10:00-14:00',
    '14:00-18:00',
    '18:00-22:00',
    '22:00-6:00 (next day)'
]
time_period_ranges = [
    '6:00-10:00',
    '10:00-14:00',
    '14:00-18:00',
    '18:00-22:00',
    '22:00-6:00 (next day)'
]
demand_per_period = [18, 20, 19, 17, 12]  # [R1, R2, R3, R4, R5]

num_shifts = 5
shift_length_hours = 8
shift_time_ranges = [
    '2:00-10:00',
    '6:00-14:00',
    '10:00-18:00',
    '14:00-22:00',
    '18:00-2:00 (next day)'
]
shifts_per_week_per_nurse = 5
work_days_per_cycle = 5
rest_days_per_cycle = 2
total_cycle_days = 7
objective_type = ['minimize_nurses']

Table_1_C_2_time_period_labels = [
    '6:00-10:00',
    '10:00-14:00',
    '14:00-18:00',
    '18:00-22:00',
    '22:00-6:00 (next day)'
]
Table_1_C_2_nurses_required = [18, 20, 19, 17, 12]

# Map to R_1..R_5 exactly as in the validated model
R_1, R_2, R_3, R_4, R_5 = demand_per_period

# Days index 1..7
days = range(1, total_cycle_days + 1)


def mod7(i: int) -> int:
    """
    Return index in {1,...,7} corresponding to integer i under modulo-7,
    matching the 'indices modulo 7' convention in the mathematical model.
    """
    r = i % 7
    return 7 if r == 0 else r


# -----------------------------------
# 2. Create model and decision variables
# -----------------------------------
model = gp.Model("Nurse_Scheduling_Minimize_Nurses")

# Decision variables:
# x_d = number of nurses whose 5-day work block starts on day d (d = 1..7)
x = model.addVars(days, vtype=gp.GRB.INTEGER, name="x")
for d in days:
    x[d].LB = 0  # Nonnegativity: x_d >= 0

# -----------------------------------
# 3. Objective function
# -----------------------------------
# Minimize total number of nurses
model.setObjective(gp.quicksum(x[d] for d in days), gp.GRB.MINIMIZE)

# -----------------------------------
# 4. Add all constraints (coverage)
#    Using the validated expressions with indices modulo 7
# -----------------------------------

# Coverage_P1_t: Time period 1 (6:00–10:00)
# x_t + x_{t-1} >= R_1,  t = 1,...,7 (indices modulo 7)
for t in days:
    t_0 = mod7(t)
    t_minus_1 = mod7(t - 1)
    model.addConstr(
        x[t_0] + x[t_minus_1] >= R_1,
        name=f"Coverage_P1_t{t}"
    )

# Coverage_P2_t: Time period 2 (10:00–14:00)
# x_{t-1} + x_{t-2} >= R_2,  t = 1,...,7 (indices modulo 7)
for t in days:
    t_minus_1 = mod7(t - 1)
    t_minus_2 = mod7(t - 2)
    model.addConstr(
        x[t_minus_1] + x[t_minus_2] >= R_2,
        name=f"Coverage_P2_t{t}"
    )

# Coverage_P3_t: Time period 3 (14:00–18:00)
# x_{t-2} + x_{t-3} >= R_3,  t = 1,...,7 (indices modulo 7)
for t in days:
    t_minus_2 = mod7(t - 2)
    t_minus_3 = mod7(t - 3)
    model.addConstr(
        x[t_minus_2] + x[t_minus_3] >= R_3,
        name=f"Coverage_P3_t{t}"
    )

# Coverage_P4_t: Time period 4 (18:00–22:00)
# x_{t-3} + x_{t-4} >= R_4,  t = 1,...,7 (indices modulo 7)
for t in days:
    t_minus_3 = mod7(t - 3)
    t_minus_4 = mod7(t - 4)
    model.addConstr(
        x[t_minus_3] + x[t_minus_4] >= R_4,
        name=f"Coverage_P4_t{t}"
    )

# Coverage_P5_t: Time period 5 (22:00–6:00 next day)
# x_{t-4} + x_{t+1} >= R_5,  t = 1,...,7 (indices modulo 7)
for t in days:
    t_minus_4 = mod7(t - 4)
    t_plus_1 = mod7(t + 1)
    model.addConstr(
        x[t_minus_4] + x[t_plus_1] >= R_5,
        name=f"Coverage_P5_t{t}"
    )

# (No indicator constraints are needed in this model.)

# -----------------------------------
# 5. Solve the model
# -----------------------------------
model.optimize()

# -----------------------------------
# 6. Print results
# -----------------------------------
if model.status == gp.GRB.OPTIMAL:
    total_nurses = model.objVal
    print("Optimal solution found.")
    print(f"Minimum number of nurses: {int(round(total_nurses))}")
    print("Start-day pattern (x_d = nurses starting work on day d):")
    for d in days:
        val = x[d].X
        if val > 1e-6:
            print(f"  Day {d}: {int(round(val))}")
else:
    print(f"Optimization ended with status code: {model.status}")
    total_nurses = float('nan')

# -----------------------------------
# 7. Output the final numerical answer in the requested format
#    The question asks: “You only need to give the minimum number of nurses.”
# -----------------------------------
print(f"FinalAnswer=【{int(round(total_nurses))}】")