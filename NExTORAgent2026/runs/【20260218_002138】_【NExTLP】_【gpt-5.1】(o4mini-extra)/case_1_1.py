import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters (given)
# =========================
HeadNurseExperienceYears = 15
ArrivalHour = 5
NumShifts = 5
ShiftLengthHours = 8
ShiftTimes = ['2:00-10:00', '6:00-14:00', '10:00-18:00', '14:00-22:00', '18:00-2:00 (next day)']
ShiftsPerWeek = 5
ConsecutiveWorkDays = 5
RestDays = 2
NursesRequired = {
    '6:00-10:00': 18,
    '10:00-14:00': 20,
    '14:00-18:00': 19,
    '18:00-22:00': 17,
    '22:00-6:00 (next day)': 12
}

# Derived for convenience
req_6_10 = NursesRequired['6:00-10:00']
req_10_14 = NursesRequired['10:00-14:00']
req_14_18 = NursesRequired['14:00-18:00']
req_18_22 = NursesRequired['18:00-22:00']
req_22_6  = NursesRequired['22:00-6:00 (next day)']

# Days indexed as 1..7, use modulo-7 arithmetic via helper function
days = range(1, 8)

def mod7(idx):
    """
    Map integer idx to day in {1,...,7} using modulo-7 arithmetic,
    consistent with x_0 = x_7, x_8 = x_1, etc.
    """
    return ((idx - 1) % 7) + 1

# =========================
# 2. Create model
# =========================
model = gp.Model("Nurse_Scheduling_Min_Staff")

# =========================
# 3. Decision variables
# =========================
# x_d: number of nurses whose first working day in the 5-day cycle is day d
x = model.addVars(days, vtype=GRB.INTEGER, lb=0, name="x")

# =========================
# 4. Objective: minimize total nurses
# =========================
model.setObjective(gp.quicksum(x[d] for d in days), GRB.MINIMIZE)

# =========================
# 5. Constraints
# =========================

# For each day t, apply coverage constraints using modulo-7 indexing
for t in days:
    t_0   = mod7(t)        # t
    t_m1  = mod7(t - 1)    # t-1
    t_m2  = mod7(t - 2)    # t-2
    t_m3  = mod7(t - 3)    # t-3
    t_m4  = mod7(t - 4)    # t-4
    t_p1  = mod7(t + 1)    # t+1

    # 6:00–10:00 on day t: x_t + x_{t-1} >= 18
    model.addConstr(x[t_0] + x[t_m1] >= req_6_10,
                    name=f"cov_6_10_day{t}")

    # 10:00–14:00 on day t: x_{t-1} + x_{t-2} >= 20
    model.addConstr(x[t_m1] + x[t_m2] >= req_10_14,
                    name=f"cov_10_14_day{t}")

    # 14:00–18:00 on day t: x_{t-2} + x_{t-3} >= 19
    model.addConstr(x[t_m2] + x[t_m3] >= req_14_18,
                    name=f"cov_14_18_day{t}")

    # 18:00–22:00 on day t: x_{t-3} + x_{t-4} >= 17
    model.addConstr(x[t_m3] + x[t_m4] >= req_18_22,
                    name=f"cov_18_22_day{t}")

    # 22:00–06:00 starting day t: x_{t-4} + x_{t+1} >= 12
    model.addConstr(x[t_m4] + x[t_p1] >= req_22_6,
                    name=f"cov_22_6_day{t}")

# No indicator constraints are required for this model.

# =========================
# 6. Solve model
# =========================
model.optimize()

# =========================
# 7. Print results
# =========================
if model.Status == GRB.OPTIMAL:
    print("Optimal solution found.")
    print(f"Minimum number of nurses (objective value): {model.ObjVal}")
    for d in days:
        print(f"x_{d} (nurses starting on day {d}): {int(round(x[d].X))}")
    min_nurses = int(round(model.ObjVal))
else:
    print(f"Optimization ended with status {model.Status}")
    min_nurses = None

# Final required output
print(f"FinalAnswer=【{min_nurses}】")