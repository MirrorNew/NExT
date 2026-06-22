def optimize_taxi_fleet():
    from gurobipy import Model, GRB
    import math

    # Create a new model
    m = Model("TaxiFleetOptimization")

    # Decision variables
    x = m.addVar(vtype=GRB.INTEGER, name="motorcycles", lb=0)
    y = m.addVar(vtype=GRB.INTEGER, name="sedans", lb=0)

    # Set objective: maximize total earnings
    m.setObjective(100 * x + 225 * y, GRB.MAXIMIZE)

    # Add constraints
    # ❤ Non-linearity is introduced. ❤
    # Capacity constraint
    # m.addConstr(30 * x + 70 * y >= 1200, name="capacity")

    # New capacity constraints with monthly cosine-based fluctuation
    # For each month t = 1,...,12: 30x + 70y >= 1200 * [1 + 0.1 * cos(pi * t / 6)]
    for t in range(1, 13):
        rhs = 1200 * (1 + 0.1 * math.cos(math.pi * t / 6.0))
        m.addConstr(30 * x + 70 * y >= rhs, name=f"capacity_month_{t}")

    # Motorcycle proportion constraint
    m.addConstr(4 * x <= y, name="motorcycle_ratio")

    # Pollution constraint
    m.addConstr(4 * x + 15 * y <= 200, name="pollution")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal objective value and decision variables
        return m.objVal, x.X, y.X
    else:
        # No feasible solution found
        return None, None, None


# Example usage
if __name__ == "__main__":
    max_earnings, opt_x, opt_y = optimize_taxi_fleet()
    if max_earnings is not None:
        print(f"Maximum Total Earnings per shift: ${max_earnings}")
        print(f"Optimal number of motorcycles (x): {opt_x}")
        print(f"Optimal number of sedans (y): {opt_y}")
    else:
        print("No feasible solution found.")