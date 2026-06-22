import gurobipy as gp
from gurobipy import GRB

# We build the MILP exactly on the validated model and then compute the monthly salary.
# FinalAnswer is the total salary to be paid in a month.


def main():
    # -----------------------------
    # 1. Parameters (from Parameters List)
    # -----------------------------
    num_guards_total = 8
    num_team_leaders = 1
    num_team_members = 7
    weeks_per_month = 4
    days_per_week = 7
    shifts_per_day = 2
    min_guards_per_shift = 3
    max_guards_per_shift = 4
    leader_max_days_per_week = 7
    member_max_days_per_week = 6
    max_shifts_per_day_per_guard = 1
    max_sitting_shifts_per_week_per_guard = 3
    min_patrol_per_shift = 1
    min_onduty_per_shift = 1
    min_business_per_shift_weekday_non_fri = 1
    min_business_per_shift_fri_weekend_day = 2
    min_business_per_night_shift = 1
    max_patrol_plus_onduty_per_guard_per_week = 4
    min_full_weekends_off_per_guard_per_month = 1
    day_shift_pay_mon_thu = 30
    night_shift_pay_mon_thu = 37
    day_shift_pay_fri_weekend = 40
    night_shift_pay_fri_weekend = 47
    patrol_bonus_per_shift = 7
    leader_bonus_per_week = 97
    required_shifts_per_week = 14
    required_guard_shifts_per_week_min = 42
    max_guard_shifts_per_week_from_day_limits = 49
    max_guard_shifts_per_week_from_sitting_limits = 24
    feasible_schedule_exists = 0
    total_salary_per_month_defined = 0
    shift_type_min_max_staff_and_roles = [
        {
            'shift_type': 'day',
            'min_total_staff': 3,
            'max_total_staff': 4,
            'min_patrol': 1,
            'min_onduty': 1,
            'min_business_non_fri_weekday': 1,
            'min_business_fri_weekend_day': 2,
        },
        {
            'shift_type': 'night',
            'min_total_staff': 3,
            'max_total_staff': 4,
            'min_patrol': 1,
            'min_onduty': 1,
            'min_business_any_night': 1,
        },
    ]
    weekly_capacity_vs_demand_summary = [
        {'description': 'required_guard_shifts_per_week_min', 'value': 42},
        {'description': 'max_guard_shifts_from_day_limits', 'value': 49},
        {'description': 'max_guard_shifts_from_sitting_limits', 'value': 24},
    ]

    # -----------------------------
    # 2. Sets / Indices
    # -----------------------------
    guards = ["A", "B", "C", "D", "E", "F", "G", "H"]
    leader = "A"
    members = ["B", "C", "D", "E", "F", "G", "H"]

    weeks = list(range(1, weeks_per_month + 1))
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    shifts = ["Day", "Night"]
    roles = ["Patrol", "Duty", "Business"]

    weekday_non_fri = ["Mon", "Tue", "Wed", "Thu"]
    fri = "Fri"
    weekend = ["Sat", "Sun"]

    # -----------------------------
    # 3. Create model
    # -----------------------------
    model = gp.Model("CommunitySecurityScheduling")

    # -----------------------------
    # 4. Decision variables
    # -----------------------------
    # x[i,w,d,s] = 1 if guard i works shift s on day d of week w
    x = model.addVars(
        guards, weeks, days, shifts,
        vtype=GRB.BINARY,
        name="x"
    )

    # y[i,w,d,s,r] = 1 if guard i is assigned role r in (w,d,s)
    y = model.addVars(
        guards, weeks, days, shifts, roles,
        vtype=GRB.BINARY,
        name="y"
    )

    # z[i,w] = 1 if guard i has a full weekend (Sat & Sun) off in week w
    z = model.addVars(
        guards, weeks,
        vtype=GRB.BINARY,
        name="z"
    )

    # -----------------------------
    # 5. Objective: minimize total number of shifts
    # -----------------------------
    model.setObjective(
        gp.quicksum(x[i, w, d, s] for i in guards for w in weeks for d in days for s in shifts),
        GRB.MINIMIZE
    )

    # -----------------------------
    # 6. Constraints
    # -----------------------------

    # (1) Single shift per day per guard
    for i in guards:
        for w in weeks:
            for d in days:
                model.addConstr(
                    gp.quicksum(x[i, w, d, s] for s in shifts) <= max_shifts_per_day_per_guard,
                    name=f"single_shift_per_day_{i}_{w}_{d}"
                )

    # (2) Weekly working days for team members: ≤ 6
    for i in members:
        for w in weeks:
            model.addConstr(
                gp.quicksum(x[i, w, d, s] for d in days for s in shifts) <= member_max_days_per_week,
                name=f"member_days_per_week_{i}_{w}"
            )

    # (3) Weekly working days for team leader: ≤ 7
    # (the sentence says 'needs to work seven days'; with 1 shift per day
    # this is represented as at most 7 shifts per week)
    for w in weeks:
        model.addConstr(
            gp.quicksum(x[leader, w, d, s] for d in days for s in shifts) <= leader_max_days_per_week,
            name=f"leader_days_per_week_{w}"
        )

    # (4) Sitting (Duty) shifts limit per week: ≤ 3
    for i in guards:
        for w in weeks:
            model.addConstr(
                gp.quicksum(y[i, w, d, s, "Duty"] for d in days for s in shifts)
                <= max_sitting_shifts_per_week_per_guard,
                name=f"sitting_limit_{i}_{w}"
            )

    # (5) Minimum personnel per shift: ≥ 3
    for w in weeks:
        for d in days:
            for s in shifts:
                model.addConstr(
                    gp.quicksum(x[i, w, d, s] for i in guards) >= min_guards_per_shift,
                    name=f"min_staff_{w}_{d}_{s}"
                )

    # (6) Maximum personnel per shift: ≤ 4
    for w in weeks:
        for d in days:
            for s in shifts:
                model.addConstr(
                    gp.quicksum(x[i, w, d, s] for i in guards) <= max_guards_per_shift,
                    name=f"max_staff_{w}_{d}_{s}"
                )

    # (7) Role assignment consistency: roles only if working
    for i in guards:
        for w in weeks:
            for d in days:
                for s in shifts:
                    model.addConstr(
                        gp.quicksum(y[i, w, d, s, r] for r in roles) <= x[i, w, d, s],
                        name=f"role_consistency_{i}_{w}_{d}_{s}"
                    )

    # (8) Exactly 1 patrol per shift
    for w in weeks:
        for d in days:
            for s in shifts:
                model.addConstr(
                    gp.quicksum(y[i, w, d, s, "Patrol"] for i in guards) == min_patrol_per_shift,
                    name=f"exactly_one_patrol_{w}_{d}_{s}"
                )

    # (9) Exactly 1 on-duty per shift
    for w in weeks:
        for d in days:
            for s in shifts:
                model.addConstr(
                    gp.quicksum(y[i, w, d, s, "Duty"] for i in guards) == min_onduty_per_shift,
                    name=f"exactly_one_duty_{w}_{d}_{s}"
                )

    # (10) Business ≥1 on Mon–Thu (all shifts)
    for w in weeks:
        for d in weekday_non_fri:
            for s in shifts:
                model.addConstr(
                    gp.quicksum(y[i, w, d, s, "Business"] for i in guards)
                    >= min_business_per_shift_weekday_non_fri,
                    name=f"business_mon_thu_{w}_{d}_{s}"
                )

    # (11) Business = 2 on Friday day shift
    for w in weeks:
        model.addConstr(
            gp.quicksum(y[i, w, fri, "Day", "Business"] for i in guards)
            == min_business_per_shift_fri_weekend_day,
            name=f"business_fri_day_{w}"
        )

    # (12) Business = 2 on weekend day shifts (Sat, Sun)
    for w in weeks:
        for d in weekend:
            model.addConstr(
                gp.quicksum(y[i, w, d, "Day", "Business"] for i in guards)
                == min_business_per_shift_fri_weekend_day,
                name=f"business_weekend_day_{w}_{d}"
            )

    # (13) Business ≥1 on all night shifts
    for w in weeks:
        for d in days:
            model.addConstr(
                gp.quicksum(y[i, w, d, "Night", "Business"] for i in guards)
                >= min_business_per_night_shift,
                name=f"business_night_{w}_{d}"
            )

    # (14) Full weekend off definition using indicator constraints:
    # If z[i,w] == 1 then all weekend shifts (Sat/Sun, Day/Night) for guard i in week w must be 0
    for i in guards:
        for w in weeks:
            model.addGenConstrIndicator(
                z[i, w], 1,
                x[i, w, "Sat", "Day"]
                + x[i, w, "Sat", "Night"]
                + x[i, w, "Sun", "Day"]
                + x[i, w, "Sun", "Night"],
                GRB.EQUAL,
                0.0,
                name=f"full_weekend_indicator_{i}_{w}"
            )

    # (15) At least one full weekend off per guard per month
    for i in guards:
        model.addConstr(
            gp.quicksum(z[i, w] for w in weeks) >= min_full_weekends_off_per_guard_per_month,
            name=f"at_least_one_full_weekend_{i}"
        )

    # (16) Patrol count limit per week: ≤ 4 per guard
    # (we keep exactly the validated mathematical form: patrol-only)
    for i in guards:
        for w in weeks:
            model.addConstr(
                gp.quicksum(y[i, w, d, s, "Patrol"] for d in days for s in shifts)
                <= max_patrol_plus_onduty_per_guard_per_week,
                name=f"patrol_limit_{i}_{w}"
            )

    # -----------------------------
    # 7. Solve the model
    # -----------------------------
    model.optimize()

    if model.status != GRB.OPTIMAL:
        # If infeasible or other status, just report and set salary to 0
        print(f"Model status: {model.status}")
        total_salary = 0.0
        print(f"FinalAnswer=【{total_salary}】")
        return

    # -----------------------------
    # Salary calculation (post-optimization)
    # -----------------------------
    def base_pay_for_shift(day, shift):
        if day in weekday_non_fri:
            if shift == "Day":
                return day_shift_pay_mon_thu
            else:
                return night_shift_pay_mon_thu
        else:  # Fri, Sat, Sun
            if shift == "Day":
                return day_shift_pay_fri_weekend
            else:
                return night_shift_pay_fri_weekend

    # Total base pay
    total_base_pay = 0.0
    for i in guards:
        for w in weeks:
            for d in days:
                for s in shifts:
                    if x[i, w, d, s].X > 0.5:
                        total_base_pay += base_pay_for_shift(d, s)

    # Patrol bonuses
    total_patrol_bonus = 0.0
    for i in guards:
        for w in weeks:
            for d in days:
                for s in shifts:
                    if y[i, w, d, s, "Patrol"].X > 0.5:
                        total_patrol_bonus += patrol_bonus_per_shift

    # Leader weekly bonus (one leader, all weeks)
    total_leader_bonus = leader_bonus_per_week * weeks_per_month

    total_salary = total_base_pay + total_patrol_bonus + total_leader_bonus

    # Optional prints
    print("Total base pay for the month:", total_base_pay)
    print("Total patrol bonus for the month:", total_patrol_bonus)
    print("Total leader bonus for the month:", total_leader_bonus)
    print("Total salary to be paid in a month:", total_salary)

    # Final required output
    print(f"FinalAnswer=【{total_salary}】")


if __name__ == "__main__":
    main()