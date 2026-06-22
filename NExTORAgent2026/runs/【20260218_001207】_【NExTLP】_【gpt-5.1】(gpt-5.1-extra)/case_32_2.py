import gurobipy as gp

# Solve the staffing optimization problem for the parcel sorting department
if __name__ == "__main__":
    # -----------------------------
    # 1. Create model
    # -----------------------------
    model = gp.Model("Express_Sorting_Staffing")

    # -----------------------------
    # 2. Parameters (STRICTLY from Parameters List)
    # -----------------------------
    # From Parameters List
    num_sorting_machines_department = 11
    machine_capacity_per_hour = 500
    full_time_daily_wage = 150
    part_time_daily_wage = 80

    # Time horizon hours (from Table_2_C12_WorkTimeChart_hours)
    hours = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    # Arrivals per time period (from Table_1_C33_Arrivals in Parameters List)
    # Map each time period to the hour index representing "up to that time"
    # We introduce hour 9 as the time just before 10:00 for "Before 10:00".
    arrivals_raw = {
        9: 5000,   # Before 10:00
        10: 4000,  # 10:00–11:00
        11: 3000,  # 11:00–12:00
        12: 4000,  # 12:00–13:00
        13: 2500,  # 13:00–14:00
        14: 3000,  # 14:00–15:00
        15: 4000,  # 15:00–16:00
        16: 4500,  # 16:00–17:00
        17: 3500,  # 17:00–18:00
        18: 2500   # 18:00–19:00
    }

    # According to the narrative, the center handles about 35,000 parcels per day.
    # We keep the total as given (avg_daily_parcels_sorting_center).
    total_parcels = 35000

    # Build cumulative arrivals A_h for h = 9..20.
    # For 9..18 use the raw arrivals; for 19,20 we cap by total_parcels to be consistent
    A = {}
    cum = 0
    for h in range(9, 19):  # 9..18
        cum += arrivals_raw[h]
        A[h] = cum

    # For hours 19 and 20, arrival is at most total_parcels
    # (this keeps C_h ≤ A_h compatible with C_20 = total_parcels)
    for h in [19, 20]:
        A[h] = total_parcels

    # Deadlines based on Parameters List description
    # Before 12:00 (i.e., parcels that have arrived by end of 11:00) must be finished before 14:00
    A_before_12 = arrivals_raw[9] + arrivals_raw[10] + arrivals_raw[11]  # 5000 + 4000 + 3000 = 12000

    # Before 15:00 (i.e., parcels that have arrived by end of 14:00) must be finished before 17:00
    A_before_15 = (arrivals_raw[9] + arrivals_raw[10] + arrivals_raw[11] +
                   arrivals_raw[12] + arrivals_raw[13] + arrivals_raw[14])  # 21500

    # -----------------------------
    # 3. Decision variables
    # -----------------------------
    # Full-time employees: x1, x2, x3
    x_1 = model.addVar(vtype=gp.GRB.INTEGER, name="x_1")  # 10:00–18:00
    x_2 = model.addVar(vtype=gp.GRB.INTEGER, name="x_2")  # 11:00–19:00
    x_3 = model.addVar(vtype=gp.GRB.INTEGER, name="x_3")  # 12:00–20:00

    # Part-time employees: y4, y5, y6
    y_4 = model.addVar(vtype=gp.GRB.INTEGER, name="y_4")  # 13:00–18:00
    y_5 = model.addVar(vtype=gp.GRB.INTEGER, name="y_5")  # 14:00–19:00
    y_6 = model.addVar(vtype=gp.GRB.INTEGER, name="y_6")  # 15:00–20:00

    # Total employees working at hour h, E_h
    E = {h: model.addVar(vtype=gp.GRB.INTEGER, name=f"E_{h}") for h in hours}

    # Parcels processed during hour h, P_h
    P = {h: model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name=f"P_{h}") for h in hours}

    # Cumulative processed parcels up to hour h, C_h
    C = {h: model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name=f"C_{h}") for h in hours}
    # C_9 is treated as parameter 0
    C_9 = 0.0

    # -----------------------------
    # 4. Constraints
    # -----------------------------

    # Employee-hour definitions (from context, exactly as given)
    model.addConstr(E[10] == x_1, name="Employee_Hour_Definition_10")
    model.addConstr(E[11] == x_1 + x_2, name="Employee_Hour_Definition_11")
    model.addConstr(E[12] == x_1 + x_2 + x_3, name="Employee_Hour_Definition_12")
    model.addConstr(E[13] == x_1 + x_2 + x_3 + y_4, name="Employee_Hour_Definition_13")
    model.addConstr(E[14] == x_1 + x_2 + x_3 + y_4 + y_5, name="Employee_Hour_Definition_14")
    model.addConstr(E[15] == x_1 + x_2 + x_3 + y_4 + y_5 + y_6, name="Employee_Hour_Definition_15")
    model.addConstr(E[16] == x_1 + x_2 + x_3 + y_4 + y_5 + y_6, name="Employee_Hour_Definition_16")
    model.addConstr(E[17] == x_1 + x_2 + x_3 + y_4 + y_5 + y_6, name="Employee_Hour_Definition_17")
    model.addConstr(E[18] == x_2 + x_3 + y_4 + y_5, name="Employee_Hour_Definition_18")
    model.addConstr(E[19] == x_2 + x_3 + y_5, name="Employee_Hour_Definition_19")
    model.addConstr(E[20] == x_3 + y_6, name="Employee_Hour_Definition_20")

    # Machine count and employee limit: E_h ≤ 11
    for h in hours:
        model.addConstr(
            E[h] <= num_sorting_machines_department,
            name=f"Machine_Count_and_Employee_Limit_{h}"
        )

    # Per-machine processing capacity: P_h ≤ 500 · E_h
    for h in hours:
        model.addConstr(
            P[h] <= machine_capacity_per_hour * E[h],
            name=f"Per_Machine_Processing_Capacity_{h}"
        )

    # Cumulative processing definition: C_h = C_{h-1} + P_h, with C_9 = 0
    prev_C_expr = C_9
    for h in hours:
        model.addConstr(
            C[h] == prev_C_expr + P[h],
            name=f"Processing_Cumulative_Definition_{h}"
        )
        prev_C_expr = C[h]

    # Cumulative processing cannot exceed cumulative arrivals: C_h ≤ A_h
    for h in hours:
        model.addConstr(
            C[h] <= A[h],
            name=f"Cumulative_Processing_le_Cumulative_Arrivals_{h}"
        )

    # Deadline: parcels arriving before 12:00 must be processed before 14:00
    model.addConstr(
        C[14] >= A_before_12,
        name="Deadline_12_to_14"
    )

    # Deadline: parcels arriving before 15:00 must be processed before 17:00
    model.addConstr(
        C[17] >= A_before_15,
        name="Deadline_15_to_17"
    )

    # End-of-day completion: all parcels must be processed before 20:00
    model.addConstr(
        C[20] == total_parcels,
        name="End_of_Day_Completion"
    )

    # Nonnegativity for P_h is already enforced by lb=0; integrality by vtype

    # -----------------------------
    # 5. Objective function
    # -----------------------------
    # Minimize total wage cost:
    # Z = 150·(x_1 + x_2 + x_3) + 80·(y_4 + y_5 + y_6)
    model.setObjective(
        full_time_daily_wage * (x_1 + x_2 + x_3) +
        part_time_daily_wage * (y_4 + y_5 + y_6),
        gp.GRB.MINIMIZE
    )

    # -----------------------------
    # 6. Solve the model
    # -----------------------------
    model.optimize()

    # -----------------------------
    # 7. Print results
    # -----------------------------
    if model.status == gp.GRB.OPTIMAL:
        print("Optimal solution found.")
        print(f"Minimum total wage cost: {model.ObjVal:.2f}")

        print("\nFull-time employees:")
        print(f"  x_1 (10:00–18:00): {int(round(x_1.X))}")
        print(f"  x_2 (11:00–19:00): {int(round(x_2.X))}")
        print(f"  x_3 (12:00–20:00): {int(round(x_3.X))}")

        print("\nPart-time employees:")
        print(f"  y_4 (13:00–18:00): {int(round(y_4.X))}")
        print(f"  y_5 (14:00–19:00): {int(round(y_5.X))}")
        print(f"  y_6 (15:00–20:00): {int(round(y_6.X))}")

        print("\nHour-by-hour staffing and processing:")
        for h in hours:
            print(
                f"Hour {h}: E_h={int(round(E[h].X))}, "
                f"P_h={P[h].X:.1f}, C_h={C[h].X:.1f}"
            )

        # Question asks: How many full-time and part-time employees should be equipped
        # to minimize total expenditure while completing tasks.
        # We output total number of employees (full-time + part-time) as the FinalAnswer.
        total_full_time = int(round(x_1.X + x_2.X + x_3.X))
        total_part_time = int(round(y_4.X + y_5.X + y_6.X))
        # You may choose what "FinalAnswer" represents; here we choose the minimum total wage cost.
        the_question_answer = model.ObjVal
    else:
        print(f"Model status: {model.status}. No optimal solution found.")
        the_question_answer = float("nan")

    # Required final statement:
    print(f"FinalAnswer=【{the_question_answer}】")