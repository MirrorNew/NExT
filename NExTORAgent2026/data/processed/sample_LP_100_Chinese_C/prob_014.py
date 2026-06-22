def optimize_transportation():
    from gurobipy import Model, GRB

    # Parameters
    total_produce = 1000
    pollution_limit = 1000
    min_horse_trips = 8
    capacity_horse = 55
    capacity_bicycle = 30
    capacity_handcart = 40
    pollution_horse = 80
    extra_pollution_per_trip_after_12 = 20
    threshold_horse_trips = 12
    M = 1000  # Large number for linking constraints

    # Create model
    m = Model("FarmerTransport")

    # Decision variables
    x_H = m.addVar(vtype=GRB.INTEGER, lb=min_horse_trips, name="x_H")
    x_B = m.addVar(vtype=GRB.INTEGER, lb=0, name="x_B")
    x_C = m.addVar(vtype=GRB.INTEGER, lb=0, name="x_C")
    y_B = m.addVar(vtype=GRB.BINARY, name="y_B")
    y_C = m.addVar(vtype=GRB.BINARY, name="y_C")

    # ❤ Non-linearity is introduced. ❤
    # Extra pollution from horse trips beyond 12 (modeled linearly with an auxiliary variable)
    z_H = m.addVar(vtype=GRB.INTEGER, lb=0, name="z_H")  # number of horse trips beyond 12
    y_H_over = m.addVar(vtype=GRB.BINARY, name="y_H_over")  # 1 if x_H > 12, else 0

    m.update()

    # ❤ Non-linearity is introduced. ❤
    # Original (commented out) objective:
    # m.setObjective(pollution_horse * x_H, GRB.MINIMIZE)
    # New objective: include extra pollution after the 12th horse trip
    m.setObjective(
        pollution_horse * x_H + extra_pollution_per_trip_after_12 * z_H,
        GRB.MINIMIZE
    )

    # Constraints
    # Produce transportation
    m.addConstr(
        capacity_horse * x_H + capacity_bicycle * x_B + capacity_handcart * x_C
        >= total_produce, "ProduceTransport")

    # ❤ Non-linearity is introduced. ❤
    # Original (commented out) pollution limit:
    # m.addConstr(pollution_horse * x_H <= pollution_limit, "PollutionLimit")
    # New pollution limit including extra pollution after the 12th horse trip
    m.addConstr(
        pollution_horse * x_H + extra_pollution_per_trip_after_12 * z_H
        <= pollution_limit,
        "PollutionLimit"
    )

    # Mode selection
    m.addConstr(y_B + y_C == 1, "ModeSelection")

    # Linking trips to mode selection
    m.addConstr(x_B <= M * y_B, "LinkBicycle")
    m.addConstr(x_C <= M * y_C, "LinkHandcart")

    # Minimum horse trips
    m.addConstr(x_H >= min_horse_trips, "MinHorseTrips")

    # ❤ Non-linearity is introduced. ❤
    # Linearization of "extra pollution only after 12 horse trips":
    # z_H = max(0, x_H - 12), implemented via MILP constraints

    # z_H cannot exceed (x_H - 12)
    m.addConstr(z_H <= x_H - threshold_horse_trips, "zH_le_xH_minus_12")

    # If y_H_over = 0, then z_H = 0
    m.addConstr(z_H <= M * y_H_over, "zH_le_M_yHover")

    # If y_H_over = 1, then z_H >= x_H - 12
    m.addConstr(z_H >= x_H - threshold_horse_trips - M * (1 - y_H_over),
                "zH_ge_xH_minus_12_if_yHover")

    # Ensure z_H is zero when x_H <= 12:
    m.addConstr(z_H <= x_H, "zH_le_xH")
    m.addConstr(z_H <= M * y_H_over, "zH_upper_yHover")

    # Optional tightening: x_H must be > 12 if y_H_over = 1
    m.addConstr(x_H >= threshold_horse_trips + 1 - M * (1 - y_H_over),
                "xH_gt_12_if_yHover")

    # Optimize
    m.optimize()

    # Check feasibility and return result
    if m.status == GRB.OPTIMAL:
        return m.objVal
    else:
        return None


if __name__ == "__main__":
    result = optimize_transportation()
    if result is not None:
        print(f"Optimal total pollution: {result}")
    else:
        print("No feasible solution found.")