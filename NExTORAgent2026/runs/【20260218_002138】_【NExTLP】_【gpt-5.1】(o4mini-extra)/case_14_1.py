import gurobipy as gp
from gurobipy import GRB

# ==============================
# 1. PARAMETERS (from Parameters List, no modification)
# ==============================

total_security_guards = 8
team_leader_count = 1
team_member_count = 7
max_shifts_per_day_per_guard = 1
max_working_days_per_week = 6
max_working_days_per_week_leader = 7
max_shifts_per_week_per_guard = 3  # used as max weekly night shifts here
weeks_per_month = 4
days_per_week = 7
shift_types = ['Day Shift', 'Night Shift']
min_guards_per_shift = 3
max_guards_per_shift = 4
min_patrol_per_shift = 1
min_on_duty_per_shift = 1
min_business_processing_per_shift = 1
business_processing_friday_weekend_day_shift = 2
min_days_off_per_month_per_guard = 1
weekend_off_days = ['Saturday', 'Sunday']
max_role_assignments_per_guard_per_week = 4
pay_day_shift_mon2thu = 30
pay_night_shift_mon2thu = 37
pay_day_shift_fri_weekend = 40
pay_night_shift_fri_weekend = 47
pay_patrol_bonus = 7
team_leader_weekly_bonus = 97
Table_1_Shift_requirements_and_roles = [
    ['Day Shift', 3, 4, 1, 1, 1, 2],
    ['Night Shift', 3, 4, 1, 1, 1, 1]
]
Table_2_Basic_Security_Parameters = [
    ['A', 1, 7],
    ['B-H', 1, 6]
]

# ==============================
# 2. SETS AND BASIC DATA
# ==============================

# Guards: A (leader), B-H (members)
guards = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
leader = 'A'
members = [g for g in guards if g != leader]

weeks = list(range(1, weeks_per_month + 1))

# Map days to names; length = 7 must match days_per_week
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
             'Friday', 'Saturday', 'Sunday']
days = list(range(days_per_week))  # 0..6 index corresponding to day_names

# Shifts: we map to short tokens 'Day', 'Night' but keep using shift_types list values
shift_map = {'Day Shift': 'Day', 'Night Shift': 'Night'}
shifts = ['Day', 'Night']  # internal names

# Roles
roles = ['patrol', 'on-duty', 'business']

# Helper: day type sets
weekday_indices = [0, 1, 2, 3]          # Mon-Thu
friday_index = 4                        # Fri
weekend_indices = [5, 6]               # Sat, Sun
fri_weekend_indices = [4, 5, 6]        # Fri, Sat, Sun

# ==============================
# 3. CREATE MODEL
# ==============================

model = gp.Model("Community_Security_Scheduling")

# ==============================
# 4. DECISION VARIABLES
# ==============================

# x[i,w,t,s] = 1 if guard i works week w, day t, shift s
x = model.addVars(
    guards, weeks, days, shifts,
    vtype=GRB.BINARY,
    name="x"
)

# y[i,w,t,s,r] = 1 if guard i on (w,t,s) does role r
y = model.addVars(
    guards, weeks, days, shifts, roles,
    vtype=GRB.BINARY,
    name="y"
)

# weekend_work[i,w] ∈ {0,1,2} = number of weekend days (Sat, Sun) guard i works in week w
weekend_work = model.addVars(
    guards, weeks,
    vtype=GRB.INTEGER,
    lb=0, ub=2,
    name="weekend_work"
)

# weekend_off[i,w] ∈ {0,1} = 1 if guard i has full weekend off in week w
weekend_off = model.addVars(
    guards, weeks,
    vtype=GRB.BINARY,
    name="weekend_off"
)

# ==============================
# 5. OBJECTIVE: Minimize total number of shifts in the month
# ==============================

model.setObjective(
    gp.quicksum(x[i, w, t, s] for i in guards for w in weeks for t in days for s in shifts),
    GRB.MINIMIZE
)

# ==============================
# 6. CONSTRAINTS
# ==============================

# C1_Daily-One-Shift: each guard at most one shift per day
for i in guards:
    for w in weeks:
        for t in days:
            model.addConstr(
                gp.quicksum(x[i, w, t, s] for s in shifts) <= max_shifts_per_day_per_guard,
                name=f"C1_DailyOneShift_{i}_{w}_{t}"
            )

# C2_Weekly-Working-Days_Members: members ≤ 6 days per week (one shift per day max)
for i in members:
    for w in weeks:
        model.addConstr(
            gp.quicksum(x[i, w, t, s] for t in days for s in shifts) <= max_working_days_per_week,
            name=f"C2_WeeklyDaysMembers_{i}_{w}"
        )

# C3_Weekly-Working-Days_TeamLeader: leader ≤ 7 days per week
for w in weeks:
    model.addConstr(
        gp.quicksum(x[leader, w, t, s] for t in days for s in shifts) <= max_working_days_per_week_leader,
        name=f"C3_WeeklyDaysLeader_{w}"
    )

# C4_Weekly-Night-Shifts: each guard ≤ 3 night shifts per week
for i in guards:
    for w in weeks:
        model.addConstr(
            gp.quicksum(x[i, w, t, 'Night'] for t in days) <= max_shifts_per_week_per_guard,
            name=f"C4_WeeklyNight_{i}_{w}"
        )

# C5_Shift-Coverage_Min: each shift needs at least 3 guards
for w in weeks:
    for t in days:
        for s in shifts:
            model.addConstr(
                gp.quicksum(x[i, w, t, s] for i in guards) >= min_guards_per_shift,
                name=f"C5_CoverageMin_{w}_{t}_{s}"
            )

# C6_Shift-Coverage_Max: each shift has at most 4 guards
for w in weeks:
    for t in days:
        for s in shifts:
            model.addConstr(
                gp.quicksum(x[i, w, t, s] for i in guards) <= max_guards_per_shift,
                name=f"C6_CoverageMax_{w}_{t}_{s}"
            )

# C7_Patrol-Role: exactly 1 patrol per shift
for w in weeks:
    for t in days:
        for s in shifts:
            model.addConstr(
                gp.quicksum(y[i, w, t, s, 'patrol'] for i in guards) == min_patrol_per_shift,
                name=f"C7_Patrol_{w}_{t}_{s}"
            )

# C8_On-Duty-Role: exactly 1 on-duty per shift
for w in weeks:
    for t in days:
        for s in shifts:
            model.addConstr(
                gp.quicksum(y[i, w, t, s, 'on-duty'] for i in guards) == min_on_duty_per_shift,
                name=f"C8_OnDuty_{w}_{t}_{s}"
            )

# C9_Business-Role-Workday/Night: at least 1 business per shift
for w in weeks:
    for t in days:
        for s in shifts:
            model.addConstr(
                gp.quicksum(y[i, w, t, s, 'business'] for i in guards) >= min_business_processing_per_shift,
                name=f"C9_BusinessMin_{w}_{t}_{s}"
            )

# C10_Business-Role-Fri/Weekend_Day: Friday & weekend day shifts need ≥2 business
for w in weeks:
    for t in fri_weekend_indices:
        s = 'Day'
        model.addConstr(
            gp.quicksum(y[i, w, t, s, 'business'] for i in guards) >= business_processing_friday_weekend_day_shift,
            name=f"C10_BusinessFriWE_{w}_{t}"
        )

# C11_Role-Link: sum of roles = x for each (i,w,t,s)
for i in guards:
    for w in weeks:
        for t in days:
            for s in shifts:
                model.addConstr(
                    gp.quicksum(y[i, w, t, s, r] for r in roles) == x[i, w, t, s],
                    name=f"C11_RoleLink_{i}_{w}_{t}_{s}"
                )

# C12_Monthly-Rest-Day: each guard works at most 27 days in month
# one shift per day, so just count shifts
for i in guards:
    model.addConstr(
        gp.quicksum(x[i, w, t, s] for w in weeks for t in days for s in shifts) <= 27,
        name=f"C12_MonthlyRest_{i}"
    )

# C13_Weekend-Work-Definition: weekend_work = number of weekend days worked
for i in guards:
    for w in weeks:
        model.addConstr(
            weekend_work[i, w] ==
            gp.quicksum(x[i, w, t, s] for t in weekend_indices for s in shifts),
            name=f"C13_WeekendWork_{i}_{w}"
        )

# C14_Weekend-Off-Link: weekend_work <= 2 * (1 - weekend_off)
for i in guards:
    for w in weeks:
        model.addConstr(
            weekend_work[i, w] <= 2 * (1 - weekend_off[i, w]),
            name=f"C14_WeekendOffLink_{i}_{w}"
        )

# C15_Minimum-One-Weekend-Off: at least one full weekend off per guard in month
for i in guards:
    model.addConstr(
        gp.quicksum(weekend_off[i, w] for w in weeks) >= 1,
        name=f"C15_AtLeastOneWeekendOff_{i}"
    )

# C16_Weekly-Role-Repetition: patrol + on-duty ≤ 4 per week per guard
for i in guards:
    for w in weeks:
        model.addConstr(
            gp.quicksum(
                y[i, w, t, s, 'patrol'] + y[i, w, t, s, 'on-duty']
                for t in days for s in shifts
            ) <= max_role_assignments_per_guard_per_week,
            name=f"C16_RoleRepetition_{i}_{w}"
        )

# Domains C17–C20 are already enforced by variable types in Gurobi

# ==============================
# 7. SOLVE MODEL
# ==============================

model.Params.OutputFlag = 0  # silent solve; remove or set to 1 to see log
model.optimize()

if model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
    print("No feasible solution found.")
    # Still obey required FinalAnswer output format; use 0 if infeasible
    print(f"FinalAnswer=【{0}】")
else:
    # ==============================
    # 8. POST-PROCESSING: COMPUTE TOTAL MONTHLY SALARY
    # ==============================

    total_salary = 0.0

    # Base shift pay + patrol bonuses
    for i in guards:
        for w in weeks:
            for t in days:
                # Day-of-week name
                day_name = day_names[t]
                is_weekend = (t in weekend_indices)
                is_friday_or_weekend = (t in fri_weekend_indices)

                for s in shifts:
                    x_val = x[i, w, t, s].X
                    if x_val < 0.5:
                        continue

                    # Determine base pay for this shift
                    if t in weekday_indices:  # Mon-Thu
                        if s == 'Day':
                            base_pay = pay_day_shift_mon2thu
                        else:
                            base_pay = pay_night_shift_mon2thu
                    else:  # Fri, Sat, Sun
                        if s == 'Day':
                            base_pay = pay_day_shift_fri_weekend
                        else:
                            base_pay = pay_night_shift_fri_weekend

                    total_salary += base_pay * x_val

                    # Patrol bonus
                    patrol_val = y[i, w, t, s, 'patrol'].X
                    total_salary += pay_patrol_bonus * patrol_val

    # Team leader weekly bonus (per week)
    total_salary += team_leader_weekly_bonus * weeks_per_month

    # Print some info (optional)
    print(f"Optimal total number of shifts (objective): {model.ObjVal}")
    print(f"Total monthly salary to be paid: {total_salary}")

    # ==============================
    # 9. REQUIRED FINAL ANSWER OUTPUT
    # ==============================
    # The question asks: "Consider how much salary is paid in a month ... Only the total salary
    # that needs to be paid needs to be given." So FinalAnswer is total_salary.
    print(f"FinalAnswer=【{total_salary}】")