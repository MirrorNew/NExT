import gurobipy as gp
from gurobipy import GRB

def solve_security_scheduling():
    # 1. Initialize model
    model = gp.Model("SecurityScheduling")

    # 2. Parameters
    num_guards = 8 # Guard 0 is Team Leader, 1-7 are Team Members
    num_weeks = 4
    num_days = 7
    num_shifts = 2  # 0: Day Shift, 1: Night Shift
    num_roles = 3   # 0: Patrol, 1: On-duty, 2: Business Processing

    # Salary related parameters
    pay_day_shift_mon2thu = 30
    pay_night_shift_mon2thu = 37
    pay_day_shift_fri_weekend = 40
    pay_night_shift_fri_weekend = 47
    pay_patrol_bonus = 7
    team_leader_weekly_bonus = 97

    # Guard role variety and fatigue parameters
    max_days_week_member = 6
    max_days_week_leader = 7
    max_on_duty_per_week = 3  # "sitting shifts... not more than 3 times a week"
    max_night_shifts_per_week = 3 # "C4_Weekly-Night-Shifts"
    max_patrol_and_on_duty_per_week = 4 # "C16_Weekly-Role-Repetition"
    max_days_per_month = 27

    # 3. Decision Variables
    # x[i, w, t, s] = 1 if guard i works in week w, day t, shift s
    x = model.addVars(num_guards, num_weeks, num_days, num_shifts, vtype=GRB.BINARY, name="x")
    # y[i, w, t, s, r] = 1 if guard i works role r in week w, day t, shift s
    y = model.addVars(num_guards, num_weeks, num_days, num_shifts, num_roles, vtype=GRB.BINARY, name="y")
    # weekend_off[i, w] = 1 if guard i has both Saturday and Sunday off in week w
    weekend_off = model.addVars(num_guards, num_weeks, vtype=GRB.BINARY, name="weekend_off")

    # 4. Objective Function: Minimize total number of shifts in the month
    model.setObjective(gp.quicksum(x[i, w, t, s] for i in range(num_guards) for w in range(num_weeks) 
                                   for t in range(num_days) for s in range(num_shifts)), GRB.MINIMIZE)

    # 5. Constraints
    for i in range(num_guards):
        # Monthly limit (C12)
        model.addConstr(gp.quicksum(x[i, w, t, s] for w in range(num_weeks) for t in range(num_days) for s in range(num_shifts)) <= max_days_per_month)
        
        # Monthly minimum one weekend off (C15)
        model.addConstr(gp.quicksum(weekend_off[i, w] for w in range(num_weeks)) >= 1)

        for w in range(num_weeks):
            # Weekly working days (C2 & C3)
            if i == 0: # Team Leader
                model.addConstr(gp.quicksum(x[i, w, t, s] for t in range(num_days) for s in range(num_shifts)) <= max_days_week_leader)
            else: # Team Members
                model.addConstr(gp.quicksum(x[i, w, t, s] for t in range(num_days) for s in range(num_shifts)) <= max_days_week_member)
            
            # Weekly night shifts (C4)
            model.addConstr(gp.quicksum(x[i, w, t, 1] for t in range(num_days)) <= max_night_shifts_per_week)
            
            # Weekly sitting (on-duty) fatigue
            model.addConstr(gp.quicksum(y[i, w, t, s, 1] for t in range(num_days) for s in range(num_shifts)) <= max_on_duty_per_week)
            
            # Weekly repetition (Patrol + On-duty <= 4) (C16)
            model.addConstr(gp.quicksum(y[i, w, t, s, 0] + y[i, w, t, s, 1] for t in range(num_days) for s in range(num_shifts)) <= max_patrol_and_on_duty_per_week)

            # Weekend off indicators (Saturday=5, Sunday=6)
            weekend_work_sum = gp.quicksum(x[i, w, t, s] for t in [5, 6] for s in range(num_shifts))
            model.addGenConstrIndicator(weekend_off[i, w], 1, weekend_work_sum == 0)
            model.addGenConstrIndicator(weekend_off[i, w], 0, weekend_work_sum >= 1)

            for t in range(num_days):
                # Max 1 shift per day (C1)
                model.addConstr(gp.quicksum(x[i, w, t, s] for s in range(num_shifts)) <= 1)
                
                for s in range(num_shifts):
                    # Role link (C11)
                    model.addConstr(gp.quicksum(y[i, w, t, s, r] for r in range(num_roles)) == x[i, w, t, s])

    for w in range(num_weeks):
        for t in range(num_days):
            for s in range(num_shifts):
                # Total personnel per shift (C5 & C6)
                model.addConstr(gp.quicksum(x[i, w, t, s] for i in range(num_guards)) >= 3)
                model.addConstr(gp.quicksum(x[i, w, t, s] for i in range(num_guards)) <= 4)
                
                # Exactly 1 patrol role (C7)
                model.addConstr(gp.quicksum(y[i, w, t, s, 0] for i in range(num_guards)) == 1)
                
                # Exactly 1 on-duty role (C8)
                model.addConstr(gp.quicksum(y[i, w, t, s, 1] for i in range(num_guards)) == 1)
                
                # Business Processing (C9 & C10)
                if s == 0 and t >= 4: # Day shift on Friday(4), Saturday(5), Sunday(6)
                    model.addConstr(gp.quicksum(y[i, w, t, s, 2] for i in range(num_guards)) == 2)
                else: # All other shifts
                    model.addConstr(gp.quicksum(y[i, w, t, s, 2] for i in range(num_guards)) >= 1)

    # 6. Solve Model
    model.setParam('OutputFlag', 0)
    model.optimize()

    # 7. Print Results
    if model.Status == GRB.OPTIMAL:
        total_salary = 0
        # Monthly base pay and patrol bonus
        for w in range(num_weeks):
            for t in range(num_days):
                for s in range(num_shifts):
                    for i in range(num_guards):
                        if x[i, w, t, s].X > 0.5:
                            if t < 4: # Monday to Thursday
                                total_salary += (pay_day_shift_mon2thu if s == 0 else pay_night_shift_mon2thu)
                            else: # Friday, Saturday, Sunday
                                total_salary += (pay_day_shift_fri_weekend if s == 0 else pay_night_shift_fri_weekend)
                            
                            if y[i, w, t, s, 0].X > 0.5: # Patrol Role Bonus
                                total_salary += pay_patrol_bonus
            # Team leader weekly bonus
            total_salary += team_leader_weekly_bonus
        
        print(f"FinalAnswer=【{int(total_salary)}】")
    else:
        print("No optimal solution found.")

if __name__ == "__main__":
    solve_security_scheduling()