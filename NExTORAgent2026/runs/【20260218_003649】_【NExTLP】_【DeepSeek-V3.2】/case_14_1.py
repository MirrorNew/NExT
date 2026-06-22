import gurobipy as gp
from gurobipy import GRB

# ======================
# 1. Parameters from Parameters List
# ======================

total_security_guards = 8
team_leader_count = 1
team_member_count = 7
max_shifts_per_day_per_guard = 1
max_working_days_per_week = 6
max_working_days_per_week_leader = 7
max_shifts_per_week_per_guard = 3
weeks_per_month = 4
days_per_week = 7
shift_types_list = ['Day Shift', 'Night Shift']
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

# Derived sets
guards = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
team_leader = ['A']
team_members = ['B', 'C', 'D', 'E', 'F', 'G', 'H']
weeks = list(range(1, weeks_per_month + 1))
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
roles = ['patrol', 'on_duty', 'business']

# Shift types simplified
shift_types = ['day', 'night']

# Weekends: Saturday, Sunday
weekend_days_list = ['Saturday', 'Sunday']

# ======================
# 2. Create Model
# ======================

model = gp.Model("Security_Scheduling")

# ======================
# 3. Decision Variables
# ======================

# x[i,w,t,s] = 1 if guard i works in week w on day t shift s
x = {}
for i in guards:
    for w in weeks:
        for t in days:
            for s in shift_types:
                x[i, w, t, s] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{w}_{t}_{s}")

# y[i,w,t,s,r] = 1 if guard i on (w,t,s) is assigned role r
y = {}
for i in guards:
    for w in weeks:
        for t in days:
            for s in shift_types:
                for r in roles:
                    y[i, w, t, s, r] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{w}_{t}_{s}_{r}")

# weekend_work[i,w] = number of weekend days (Sat, Sun) guard i works in week w
weekend_work = {}
for i in guards:
    for w in weeks:
        weekend_work[i, w] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=2, name=f"weekend_work_{i}_{w}")

# weekend_off[i,w] = 1 if guard i has both Sat and Sun off in week w
weekend_off = {}
for i in guards:
    for w in weeks:
        weekend_off[i, w] = model.addVar(vtype=GRB.BINARY, name=f"weekend_off_{i}_{w}")

# Total salary variable
total_salary = model.addVar(vtype=GRB.CONTINUOUS, name="total_salary")

model.update()

# ======================
# 4. Constraints
# ======================

# C1: Daily one shift limit
for i in guards:
    for w in weeks:
        for t in days:
            model.addConstr(
                gp.quicksum(x[i, w, t, s] for s in shift_types) <= max_shifts_per_day_per_guard,
                name=f"C1_daily_limit_{i}_{w}_{t}"
            )

# C2: Weekly working days limit for team members
for i in team_members:
    for w in weeks:
        model.addConstr(
            gp.quicksum(x[i, w, t, s] for t in days for s in shift_types) <= max_working_days_per_week,
            name=f"C2_weekly_days_member_{i}_{w}"
        )

# C3: Weekly working days limit for team leader
for w in weeks:
    model.addConstr(
        gp.quicksum(x['A', w, t, s] for t in days for s in shift_types) <= max_working_days_per_week_leader,
        name=f"C3_weekly_days_leader_{w}"
    )

# C4: Weekly night shifts limit (max_shifts_per_week_per_guard = 3)
for i in guards:
    for w in weeks:
        model.addConstr(
            gp.quicksum(x[i, w, t, 'night'] for t in days) <= max_shifts_per_week_per_guard,
            name=f"C4_night_shifts_{i}_{w}"
        )

# C5: Shift coverage min
for w in weeks:
    for t in days:
        for s in shift_types:
            model.addConstr(
                gp.quicksum(x[i, w, t, s] for i in guards) >= min_guards_per_shift,
                name=f"C5_min_coverage_{w}_{t}_{s}"
            )

# C6: Shift coverage max
for w in weeks:
    for t in days:
        for s in shift_types:
            model.addConstr(
                gp.quicksum(x[i, w, t, s] for i in guards) <= max_guards_per_shift,
                name=f"C6_max_coverage_{w}_{t}_{s}"
            )

# C7: Exactly one patrol per shift
for w in weeks:
    for t in days:
        for s in shift_types:
            model.addConstr(
                gp.quicksum(y[i, w, t, s, 'patrol'] for i in guards) == min_patrol_per_shift,
                name=f"C7_patrol_{w}_{t}_{s}"
            )

# C8: Exactly one on-duty per shift
for w in weeks:
    for t in days:
        for s in shift_types:
            model.addConstr(
                gp.quicksum(y[i, w, t, s, 'on_duty'] for i in guards) == min_on_duty_per_shift,
                name=f"C8_on_duty_{w}_{t}_{s}"
            )

# C9: At least one business role per shift
for w in weeks:
    for t in days:
        for s in shift_types:
            model.addConstr(
                gp.quicksum(y[i, w, t, s, 'business'] for i in guards) >= min_business_processing_per_shift,
                name=f"C9_business_{w}_{t}_{s}"
            )

# C10: Business role requirement for Friday/weekend day shift (>=2)
for w in weeks:
    for t in ['Friday', 'Saturday', 'Sunday']:
        model.addConstr(
            gp.quicksum(y[i, w, t, 'day', 'business'] for i in guards) >= business_processing_friday_weekend_day_shift,
            name=f"C10_fri_weekend_business_{w}_{t}"
        )

# C11: Role assignment consistent with working
for i in guards:
    for w in weeks:
        for t in days:
            for s in shift_types:
                model.addConstr(
                    gp.quicksum(y[i, w, t, s, r] for r in roles) == x[i, w, t, s],
                    name=f"C11_role_link_{i}_{w}_{t}_{s}"
                )

# C12: Monthly rest days (implied from weekly constraints, but explicitly stated)
# Members: max 6 days/week * 4 weeks = 24 working days max, so at least 4 days off (28-24=4)
# Leader: max 7 days/week * 4 weeks = 28 working days max
for i in team_members:
    model.addConstr(
        gp.quicksum(x[i, w, t, s] for w in weeks for t in days for s in shift_types) <= 24,
        name=f"C12_monthly_member_{i}"
    )
model.addConstr(
    gp.quicksum(x['A', w, t, s] for w in weeks for t in days for s in shift_types) <= 28,
    name="C12_monthly_leader_A"
)

# C13: Weekend work definition
for i in guards:
    for w in weeks:
        model.addConstr(
            weekend_work[i, w] == gp.quicksum(x[i, w, t, s] for t in weekend_days_list for s in shift_types),
            name=f"C13_weekend_work_def_{i}_{w}"
        )

# C14: Weekend off link (weekend_work <= 2*(1 - weekend_off))
for i in guards:
    for w in weeks:
        model.addConstr(
            weekend_work[i, w] <= 2 * (1 - weekend_off[i, w]),
            name=f"C14_weekend_off_link_{i}_{w}"
        )

# C15: At least one full weekend off per month
for i in guards:
    model.addConstr(
        gp.quicksum(weekend_off[i, w] for w in weeks) >= min_days_off_per_month_per_guard,
        name=f"C15_min_weekend_off_{i}"
    )

# C16: Weekly role repetition limit (patrol + on-duty ≤ 4)
for i in guards:
    for w in weeks:
        model.addConstr(
            gp.quicksum(y[i, w, t, s, 'patrol'] + y[i, w, t, s, 'on_duty'] 
                       for t in days for s in shift_types) <= max_role_assignments_per_guard_per_week,
            name=f"C16_role_repetition_{i}_{w}"
        )

# ======================
# 5. Salary Calculation
# ======================

# Base pay calculation
base_pay = gp.quicksum(
    (pay_day_shift_mon2thu if t in ['Monday', 'Tuesday', 'Wednesday', 'Thursday'] and s == 'day' else
     pay_night_shift_mon2thu if t in ['Monday', 'Tuesday', 'Wednesday', 'Thursday'] and s == 'night' else
     pay_day_shift_fri_weekend if t in ['Friday', 'Saturday', 'Sunday'] and s == 'day' else
     pay_night_shift_fri_weekend) * x[i, w, t, s]
    for i in guards for w in weeks for t in days for s in shift_types
)

# Patrol bonus
patrol_bonus = gp.quicksum(
    pay_patrol_bonus * y[i, w, t, s, 'patrol']
    for i in guards for w in weeks for t in days for s in shift_types
)

# Team leader weekly bonus (4 weeks)
leader_bonus = team_leader_weekly_bonus * weeks_per_month

# Total salary constraint
model.addConstr(
    total_salary == base_pay + patrol_bonus + leader_bonus,
    name="total_salary_calc"
)

# ======================
# 6. Objective
# ======================

model.setObjective(total_salary, GRB.MINIMIZE)

# ======================
# 7. Solve and Output
# ======================

model.optimize()

if model.status == GRB.OPTIMAL:
    total_salary_value = round(total_salary.X, 2)
    print(f"FinalAnswer=【{total_salary_value}】")
else:
    print(f"FinalAnswer=【No feasible solution】")