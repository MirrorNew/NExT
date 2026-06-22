import gurobipy as gp
from gurobipy import GRB


def solve_flight_gate_assignment():
    """
    Models and solves the flight gate assignment problem to minimize total delay cost.
    Non-linearity (Type A) is handled via linearization.
    """
    gates = [1, 2]
    flights = [1, 2, 3, 4]
    arrange_start_time = {1: 600, 2: 630, 4: 645, 3: 675}
    duration = {1: 60, 2: 60, 3: 75, 4: 60}
    cost_per_delay = 1
    big_m = 1500

    # --- 1. Model Creation ---
    model = gp.Model("FlightGateAssignment")

    # --- 2. Decision Variables ---
    # x[f,g] = 1 if flight f is assigned to gate g, else 0
    x = model.addVars([(f, g) for f in flights for g in gates], vtype=GRB.BINARY, name="x")

    # s[f] = actual start time of flight f
    s = model.addVars(flights, vtype=GRB.INTEGER, name="s")

    # ❤ Non-linearity is introduced. ❤
    # z = 1 only if F1, F2 and F3 are all assigned to gate G1 simultaneously
    z = model.addVar(vtype=GRB.BINARY, name="z_F1F2F3_G1")

    # --- 3. Objective Function ---
    # Base part: Minimize total delay cost with base cost_per_delay
    # ❤ Non-linearity is introduced. ❤
    # Original linear objective (commented out):
    # obj = gp.quicksum(cost_per_delay * (s[f] - arrange_start_time[f])
    #                   for f in flights)

    # Linearized objective with amplification when F1, F2, F3 all in G1:
    #   base cost_per_delay * delay +
    #   extra 2 * delay for flights 1,2,3 if z == 1
    base_delay_cost = gp.quicksum(cost_per_delay * (s[f] - arrange_start_time[f])  for f in flights)
    amplified_delay_cost = gp.quicksum(
        2 * (s[f] - arrange_start_time[f]) for f in [1, 2, 3]
    ) * z

    obj = base_delay_cost + amplified_delay_cost
    model.setObjective(obj, GRB.MINIMIZE)

    # --- 4. Constraints ---
    # Constraint 1: Actual start time should be greater than or equal to the scheduled start time
    for f in flights:
        model.addConstr(s[f] >= arrange_start_time[f], f"start_time_constraint_{f}")

    # Constraint 2: Every flight should be assigned to exactly one gate
    for f in flights:
        model.addConstr(gp.quicksum(x[f, g] for g in gates) == 1, f"one_gate_per_flight_{f}")

    # Constraint 3: The actual start time should be greater than or equal to the end time
    # of the previous flight at the same gate (linearized using Big-M).
    # This formulation imposes a fixed sequence f1 -> f2 if f1 < f2.
    for g in gates:
        for f1 in flights:
            for f2 in flights:
                if f1 < f2:  # for flights where f1 comes before f2
                    # If both flights are assigned to the same gate g,
                    # then f2 must start after f1 ends
                    model.addConstr(
                        s[f2] >= s[f1] + duration[f1] - big_m * (2 - x[f1, g] - x[f2, g]),
                        f"sequence_constraint_{f1}_{f2}_{g}",
                    )

    # ❤ Non-linearity is introduced. ❤
    # Linearization constraints to ensure z = x[1,1] * x[2,1] * x[3,1]
    # Let y_12 = x[1,1] * x[2,1], then z = y_12 * x[3,1]
    y_12 = model.addVar(vtype=GRB.BINARY, name="y_12_F1F2_G1")

    # y_12 = x[1,1] * x[2,1]
    model.addConstr(y_12 <= x[1, 1], "y12_le_x11")
    model.addConstr(y_12 <= x[2, 1], "y12_le_x21")
    model.addConstr(y_12 >= x[1, 1] + x[2, 1] - 1, "y12_ge_x11_x21")

    # z = y_12 * x[3,1]
    model.addConstr(z <= y_12, "z_le_y12")
    model.addConstr(z <= x[3, 1], "z_le_x31")
    model.addConstr(z >= y_12 + x[3, 1] - 1, "z_ge_y12_x31")

    # --- 5. Solve the Model ---
    # model.setParam("OutputFlag", 0) # Suppress Gurobi output
    model.optimize()

    # --- 6. Return Results ---
    if model.status == GRB.OPTIMAL:
        # Optional: you could return assignment details if needed
        result = {
            "status": "optimal",
            "obj": model.ObjVal,
            "z_value": z.X,
            "assignments": {(f, g): x[f, g].X for f in flights for g in gates},
            "start_times": {f: s[f].X for f in flights},
        }
        return result
    else:
        return {"status": f"{model.status}"}


# Run the solver function
if __name__ == "__main__":
    result = solve_flight_gate_assignment()
    print(result)