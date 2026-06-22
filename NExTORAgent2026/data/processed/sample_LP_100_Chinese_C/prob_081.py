def optimize_hydrogen_transport():
    from gurobipy import Model, GRB, quicksum

    # Create a new model
    m = Model("HydrogenTransport")

    # Decision variables: number of trips for each method
    # x: high-pressure tube trailer trips
    # y: liquefied hydrogen tanker trips
    x = m.addVar(vtype=GRB.INTEGER, name="x", lb=0)
    y = m.addVar(vtype=GRB.INTEGER, name="y", lb=0)

    # ❤ Non-linearity is introduced. ❤
    # Original (commented out) linear objective:
    # m.setObjective(x + y, GRB.MINIMIZE)

    # New objective: still minimize total trips (unchanged mathematically,
    # no non-linearity needed here), but we keep the comment to mark where
    # the change in the problem description starts to matter (in the budget).
    m.setObjective(x + y, GRB.MINIMIZE)

    # Add constraints
    # Volume constraint
    m.addConstr(50 * x + 30 * y >= 1000, name="volume_constraint")

    # ❤ Non-linearity is introduced. ❤
    # Original (commented out) budget constraint:
    # m.addConstr(500 * x + 200 * y <= 3750, name="budget_constraint")

    # To model the "if y > 8 then pay an extra fixed 400" within MILP,
    # we introduce a binary variable z and a big-M formulation to linearize
    # the non-linear conditional cost.
    #
    # Non-linear idea: budget = 500x + 200y + 400 * I(y > 8) <= 3750
    #
    # Linearization:
    #   z ∈ {0, 1}
    #   z = 1  ⇒  extra 400 cost is applied
    #   If y > 8  ⇒  z must be 1
    #   If y ≤ 8  ⇒  z can be 0 (and optimal solution will prefer z = 0)
    #
    # We use:
    #   y <= 8 + M * z
    #   y >= 9 - M * (1 - z)
    # and budget:
    #   500 x + 200 y + 400 z <= 3750

    # Binary variable indicating if the safety check fee is incurred
    z = m.addVar(vtype=GRB.BINARY, name="z")

    # Big-M constant: must be >= maximum possible number of y trips
    # Here, with budget 3750 and cost per y-trip 200,
    # y is at most 18 even without x, so M = 20 is safe.
    M = 20

    # Logical constraint: if y > 8 then z = 1 (safety fee applies)
    # Ensuring "y > 8 ⇒ z = 1" and allowing "y ≤ 8 ⇒ z = 0" (optimal will choose 0)
    m.addConstr(y <= 8 + M * z, name="upper_link_y_z")
    m.addConstr(y >= 9 - M * (1 - z), name="lower_link_y_z")

    # Updated budget constraint including the possible extra 400 fixed fee
    m.addConstr(500 * x + 200 * y + 400 * z <= 3750, name="budget_constraint")

    # Relationship constraint: x < y
    m.addConstr(x + 1 <= y, name="less_trips_constraint")  # x < y is equivalent to x + 1 <= y

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        total_trips = m.objVal
        x_val = int(x.X)
        y_val = int(y.X)
        z_val = int(z.X)
        return total_trips, x_val, y_val, z_val
    else:
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_hydrogen_transport()
    if result is not None:
        total_trips, x_val, y_val, z_val = result
        print(f"Minimum Total Trips: {total_trips}")
        print(f"High-pressure tube trailer trips (x): {x_val}")
        print(f"Liquefied hydrogen tanker trips (y): {y_val}")
        print(f"Safety check fee applied (z=1 means yes): {z_val}")
    else:
        print("No feasible solution found.")