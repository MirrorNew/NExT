import gurobipy as gp
from gurobipy import GRB


def solve_flight_gate_assignment(
    gates=[1, 2],
    flights=[1, 2, 3, 4],
    arrange_start_time={1: 600, 2: 630, 4: 645, 3: 675},
    duration={1: 60, 2: 60, 3: 75, 4: 60},
    cost_per_delay=1,
    big_m=1500,
):
    """
    Models and solves the flight gate assignment problem to minimize total delay,
    with a nonlinear penalty: if more than 2 flights are assigned to the same gate
    within 10:00-12:30 (600-750), the delay cost of all flights at that gate is
    multiplied by 1.5.
    """
    # --- 1. Model Creation ---
    model = gp.Model("FlightGateAssignment")

    # --- 2. Decision Variables ---
    # x[f,g] = 1 if flight f is assigned to gate g, else 0
    x = model.addVars([(f, g) for f in flights for g in gates],
                      vtype=GRB.BINARY, name="x")

    # s[f] = actual start time of flight f
    s = model.addVars(flights, vtype=GRB.INTEGER, name="s")

    # y[g] = 1 if gate g has more than 2 flights in the 10:00-12:30 window, else 0
    y = model.addVars(gates, vtype=GRB.BINARY, name="y")

    # Helper: list of flights that are in the 10:00-12:30 time window
    # (by scheduled start time, consistent with the problem statement)
    window_start = 600  # 10:00
    window_end = 750    # 12:30
    flights_in_window = [
        f for f in flights
        if arrange_start_time[f] >= window_start and arrange_start_time[f] <= window_end
    ]

    # --- 3. Objective Function ---

    # ❤ Non-linearity is introduced. ❤
    # Original linear objective is commented out and replaced by a quadratic one
    # that multiplies delay at a gate by (1 + 0.5 * y_g), which becomes 1.5 when y_g = 1.

    # obj = gp.quicksum(cost_per_delay * (s[f] - arrange_start_time[f])
    #                   for f in flights)

    # New (nonlinear) objective:
    # Base delay per flight:
    base_delay = {
        f: (s[f] - arrange_start_time[f])
        for f in flights
    }

    # For each gate, total delay at that gate:

    YYY = model.addVars(gates)
    for g in gates:
        model.addConstr(YYY[g] == gp.quicksum(base_delay[f] * x[f, g] for f in flights))
    gate_delay = {
        g: YYY[g]
        for g in gates
    }
    # For each gate, apply multiplier (1 + 0.5 * y[g]) to its delay:
    # This introduces bilinear terms: gate_delay[g] * y[g]
    obj = gp.quicksum(
        cost_per_delay * (gate_delay[g] + 0.5 * gate_delay[g] * y[g])
        for g in gates
    )

    model.setObjective(obj, GRB.MINIMIZE)

    # --- 4. Constraints ---
    # Constraint 1: Actual start time should be greater than or equal to
    # the scheduled start time
    for f in flights:
        model.addConstr(s[f] >= arrange_start_time[f],
                        f"start_time_constraint_{f}")

    # Constraint 2: Every flight should be assigned to exactly one gate
    for f in flights:
        model.addConstr(gp.quicksum(x[f, g] for g in gates) == 1,
                        f"one_gate_per_flight_{f}")

    # Constraint 3: No overlap of flights on the same gate (Big-M sequencing)
    for g in gates:
        for f1 in flights:
            for f2 in flights:
                if f1 < f2:
                    model.addConstr(
                        s[f2] >= s[f1] + duration[f1]
                        - big_m * (2 - x[f1, g] - x[f2, g]),
                        f"sequence_constraint_{f1}_{f2}_{g}",
                    )

    # Constraint 4: Define y[g] based on number of flights assigned to gate g
    # in the 10:00-12:30 window.
    #
    # If sum_{f in window} x[f,g] > 2, then y[g] = 1.
    # We linearize this with:
    #   sum_{f in window} x[f,g] - 3 * y[g] <= 2      (forces y[g]=1 if count>=3)
    #   y[g] <= (1/3) * sum_{f in window} x[f,g]      (forces y[g]=0 if count<=2)
    for g in gates:
        if len(flights_in_window) > 0:
            model.addConstr(
                gp.quicksum(x[f, g] for f in flights_in_window) - 3 * y[g] <= 2,
                f"y_upper_trigger_{g}",
            )
            model.addConstr(
                y[g] <= (1.0 / 3.0) * gp.quicksum(x[f, g] for f in flights_in_window),
                f"y_lower_trigger_{g}",
            )
        else:
            # No flights in window ⇒ must be 0
            model.addConstr(y[g] == 0, f"y_zero_{g}")

    # --- 5. Solve the Model ---
    # You may uncomment the next line to suppress Gurobi output:
    # model.setParam("OutputFlag", 0)
    model.optimize()

    # --- 6. Return Results ---
    if model.status == GRB.OPTIMAL:
        solution = {
            "status": "optimal",
            "obj": model.ObjVal,
            "assignment": {(f, g): int(x[f, g].X + 0.5) for f in flights for g in gates},
            "start_times": {f: int(s[f].X + 0.5) for f in flights},
            "gate_penalty_flag": {g: int(y[g].X + 0.5) for g in gates},
        }
        return solution
    else:
        return {"status": f"{model.status}"}


# Run the solver function
if __name__ == "__main__":
    result = solve_flight_gate_assignment()
    print(result)