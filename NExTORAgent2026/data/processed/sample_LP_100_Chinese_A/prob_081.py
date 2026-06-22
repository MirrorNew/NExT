def optimize_hydrogen_transport():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("HydrogenTransport")

    # Decision variables: number of trips for each method
    # x: high-pressure tube trailer trips
    # y: liquefied hydrogen tanker trips
    x = m.addVar(vtype=GRB.INTEGER, name="x", lb=2)
    y = m.addVar(vtype=GRB.INTEGER, name="y", lb=30)

    # ❤ Non-linearity is introduced. ❤
    # m.setObjective(x + y, GRB.MINIMIZE)
    # Set the new nonlinear objective: minimize 2 * x * y * (x + y)
    Y = m.addVar()
    m.addConstr(Y == x * y)
    m.setObjective(2 * Y * (x + y), GRB.MINIMIZE)

    # Add constraints
    # Volume constraint
    m.addConstr(50 * x + 30 * y >= 1000, name="volume_constraint")
    # Budget constraint
    m.addConstr(500 * x + 200 * y <= 9750, name="budget_constraint")
    # Relationship constraint: x < y
    m.addConstr(x + 1 <= y, name="less_trips_constraint")  # x < y is equivalent to x + 1 <= y

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return both the optimal trips and the minimized nonlinear effective cost
        optimal_x = int(x.X)
        optimal_y = int(y.X)
        effective_cost = m.objVal
        return optimal_x, optimal_y, effective_cost
    else:
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_hydrogen_transport()
    if result is not None:
        optimal_x, optimal_y, effective_cost = result
        print(f"Optimal high-pressure tube trailer trips (x): {optimal_x}")
        print(f"Optimal liquefied hydrogen tanker trips (y): {optimal_y}")
        print(f"Minimum nonlinear effective transport cost 2*x*y*(x+y): {effective_cost}")
    else:
        print("No feasible solution found.")