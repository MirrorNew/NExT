import gurobipy as gp
from gurobipy import GRB

# Define parameters strictly from the Parameters List
total_security_guards = 8
team_leader_count = 1
team_member_count = 7
max_shifts_per_day_per_guard = 1
max_working_days_per_week = 6
max_working_days_per_week_leader = 7
max_shifts_per_week_per_guard = 3 # Interpreted as max night shifts based on context C4
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

# Indices
# Guards: 0 is Team Leader (A), 1-7 are Members (B-H)
Guards = range(total_security_guards)
Weeks = range(weeks_per_month)
Days = range(days_per_week) # 0: Mon, 1: Tue, ..., 4: Fri, 5: Sat, 6: Sun
Shifts = [0, 1] # 0: Day Shift, 1: Night Shift
Roles = ['patrol', 'on-duty', 'business']

# Model
model = gp.Model("CommunitySecurityScheduling")

# Variables
# x[i,w,t,s] = 1 if guard i works in week w, day t, shift s
x = model.addVars(Guards, Weeks, Days, Shifts, vtype=GRB.BINARY, name="x")

# y[i,w,t,s,r] = 1 if guard i has role r in week w, day t, shift s
y = model.addVars(Guards, Weeks, Days, Shifts, Roles, vtype=GRB.BINARY, name="y")

# weekend_off[i,w] = 1 if guard i has both Sat and Sun off in week w
weekend_off = model.addVars(Guards, Weeks, vtype=GRB.BINARY, name="weekend_off")

# Objective: Minimize total number of shifts
model.setObjective(gp.quicksum(x[i, w, t, s] for i in Guards for w in Weeks for t in Days for s in Shifts), GRB.MINIMIZE)

# Constraints

# C1: Daily One Shift
model.addConstrs((gp.quicksum(x[i, w, t, s] for s in Shifts) <= max_shifts_per_day_per_guard
                  for i in Guards for w in Weeks for t in Days), "DailyOneShift")

# C2: Weekly Working Days (Members B-H)
model.addConstrs((gp.quicksum(x[i, w, t, s] for t in Days for s in Shifts) <= max_working_days_per_week
                  for i in range(1, total_security_guards) for w in Weeks), "WeeklyWorkingDays_Members")

# C3: Weekly Working Days (Team Leader A)
model.addConstrs((gp.quicksum(x[0, w, t, s] for t in Days for s in Shifts) <= max_working_days_per_week_leader
                  for w in Weeks), "WeeklyWorkingDays_Leader")

# C4: Weekly Night Shifts (Sitting Shifts) limit
# Context implies "sitting shifts" maps to Night Shift constraint
model.addConstrs((gp.quicksum(x[i, w, t, 1] for t in Days) <= max_shifts_per_week_per_guard
                  for i in Guards for w in Weeks), "WeeklyNightShifts")

# C5 & C6: Shift Coverage (Min 3, Max 4)
model.addConstrs((gp.quicksum(x[i, w, t, s] for i in Guards) >= min_guards_per_shift
                  for w in Weeks for t in Days for s in Shifts), "ShiftCoverage_Min")
model.addConstrs((gp.quicksum(x[i, w, t, s] for i in Guards) <= max_guards_per_shift
                  for w in Weeks for t in Days for s in Shifts), "ShiftCoverage_Max")

# C11: Role Link (If working, must have exactly one role; else 0)
model.addConstrs((gp.quicksum(y[i, w, t, s, r] for r in Roles) == x[i, w, t, s]
                  for i in Guards for w in Weeks for t in Days for s in Shifts), "RoleLink")

# Role Requirements per Shift
for w in Weeks:
    for t in Days:
        for s in Shifts:
            # C7: Patrol Role
            model.addConstr(gp.quicksum(y[i, w, t, s, 'patrol'] for i in Guards) == min_patrol_per_shift, 
                            f"PatrolRole_{w}_{t}_{s}")
            # C8: On-Duty Role
            model.addConstr(gp.quicksum(y[i, w, t, s, 'on-duty'] for i in Guards) == min_on_duty_per_shift, 
                            f"OnDutyRole_{w}_{t}_{s}")
            
            # C9 & C10: Business Role
            # Requirement: 2 for Day Shift (s=0) on Fri(4), Sat(5), Sun(6)
            # 1 otherwise
            req_business = min_business_processing_per_shift
            if s == 0 and t in [4, 5, 6]:
                req_business = business_processing_friday_weekend_day_shift
            
            model.addConstr(gp.quicksum(y[i, w, t, s, 'business'] for i in Guards) >= req_business, 
                            f"BusinessRole_{w}_{t}_{s}")

# C12: Monthly Rest Day
# Max shifts = 4 weeks * 7 days = 28. "At least one day in a month" means max 27 shifts.
model.addConstrs((gp.quicksum(x[i, w, t, s] for w in Weeks for t in Days for s in Shifts) <= 27
                  for i in Guards), "MonthlyRestDay")

# C13, C14, C15: Weekend Off Constraints
# C15: At least one full weekend off per month
model.addConstrs((gp.quicksum(weekend_off[i, w] for w in Weeks) >= 1
                  for i in Guards), "MinOneWeekendOff")

# C14: Link weekend_off variable to actual schedule using Indicator Constraint
# If weekend_off[i,w] == 1, then sum of x on Sat/Sun must be 0
for i in Guards:
    for w in Weeks:
        weekend_work_sum = gp.quicksum(x[i, w, t, s] for t in [5, 6] for s in Shifts)
        model.addGenConstrIndicator(weekend_off[i, w], 1, weekend_work_sum == 0, name=f"WeekendOffInd_{i}_{w}")

# C16: Weekly Role Repetition (Patrol + On-duty <= 4)
model.addConstrs((gp.quicksum(y[i, w, t, s, 'patrol'] + y[i, w, t, s, 'on-duty'] 
                              for t in Days for s in Shifts) <= max_role_assignments_per_guard_per_week
                  for i in Guards for w in Weeks), "WeeklyRoleRepetition")

# Solve
model.optimize()

# Calculate Total Salary
if model.Status == GRB.OPTIMAL:
    total_salary = 0
    for i in Guards:
        guard_pay = 0
        # Leader Bonus (only for guard 0)
        if i == 0:
            guard_pay += team_leader_weekly_bonus * weeks_per_month
            
        for w in Weeks:
            for t in Days:
                for s in Shifts:
                    if x[i, w, t, s].X > 0.5:
                        # Determine base rate
                        rate = 0
                        is_weekend_peak = (t >= 4) # Fri, Sat, Sun
                        
                        if not is_weekend_peak: # Mon-Thu
                            if s == 0: # Day
                                rate = pay_day_shift_mon2thu
                            else: # Night
                                rate = pay_night_shift_mon2thu
                        else: # Fri-Sun
                            if s == 0: # Day
                                rate = pay_day_shift_fri_weekend
                            else: # Night
                                rate = pay_night_shift_fri_weekend
                        
                        guard_pay += rate
                        
                        # Patrol Bonus
                        if y[i, w, t, s, 'patrol'].X > 0.5:
                            guard_pay += pay_patrol_bonus
        
        total_salary += guard_pay
    
    print(f"Optimal Total Shifts: {model.ObjVal}")
    print(f"FinalAnswer=【{int(total_salary)}】")
else:
    print("No feasible solution found.")
    print(f"FinalAnswer=【No Solution】")