def optimize_microcomputer_production():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("Microcomputer_Production_Nonlinear")

    # Decision variables
    # Production quantities
    x_A = m.addVar(name="x_A", lb=10)  # at least 10 units of A
    x_B = m.addVar(name="x_B", lb=15)  # at least 15 units of B

    # Overtime units produced
    x_A_OT = m.addVar(name="x_A_OT", lb=0)
    x_B_OT = m.addVar(name="x_B_OT", lb=0)

    # Overtime hours in Process II
    overtime_hours = m.addVar(name="overtime_hours", lb=0, ub=30)

    # ---------------- Objective: introduce non-linear synergy term ----------------
    # ❤ Non-linearity is introduced. ❤
    # profit = (300 * x_A + 450 * x_B - 20 * x_A_OT - 25 * x_B_OT)
    # Original linear profit kept as base part
    linear_profit = (300 * x_A + 450 * x_B - 20 * x_A_OT - 25 * x_B_OT)

    # Non-linear synergy term: 5 * x * y * z
    # Here: x = x_A, y = x_B, z = overtime_hours
    Y = m.addVar(name="Nlinear_profit", lb=0)
    m.addConstr(Y == x_A * x_B )
    synergy_profit = 0.2 * Y * overtime_hours

    # Total profit = linear profit + non-linear synergy profit
    total_profit_obj = linear_profit + synergy_profit
    m.setObjective(total_profit_obj, GRB.MAXIMIZE)

    # Constraints
    # Process I exact utilization
    m.addConstr(4 * x_A + 6 * x_B == 150, name="ProcessI_Capacity")

    # Process II regular hours
    m.addConstr(3 * x_A + 2 * x_B <= 70, name="ProcessII_Regular_Capacity")

    # Overtime hours constraint
    # Total overtime hours in Process II: 3 hours per unit of A during overtime, 2 hours per unit of B
    m.addConstr(3 * x_A_OT + 2 * x_B_OT <= 30, name="Overtime_Hours_Limit")

    # Link overtime units to total production
    m.addConstr(x_A_OT <= x_A, name="X_A_OT_Limit")
    m.addConstr(x_B_OT <= x_B, name="X_B_OT_Limit")

    # Calculate total overtime hours used
    # Total overtime hours in Process II:
    # 3 hours per unit of A during overtime, 2 hours per unit of B during overtime
    m.addConstr(overtime_hours == 3 * x_A_OT + 2 * x_B_OT,
                name="Total_Overtime_Hours")

    # Overtime hours cannot exceed 30
    m.addConstr(overtime_hours <= 30, name="Max_Overtime_Hours")

    # ---------------- Profit constraint with non-linearity ----------------
    # ❤ Non-linearity is introduced. ❤
    # total_profit = (300 * x_A + 450 * x_B - 20 * x_A_OT - 25 * x_B_OT)
    # m.addConstr(total_profit >= 10000, name="Profit_Minimum")
    # Now total profit includes the non-linear synergy term
    total_profit_constr = linear_profit + synergy_profit
    m.addConstr(total_profit_constr >= 10000, name="Profit_Minimum")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        return m.objVal
    else:
        return None


if __name__ == "__main__":
    result = optimize_microcomputer_production()
    if result is not None:
        print(f"Optimal profit (with non-linear synergy): {result}")
    else:
        print("No feasible solution found.")