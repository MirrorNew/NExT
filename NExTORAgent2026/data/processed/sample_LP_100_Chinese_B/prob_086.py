def optimize_wraps_and_platters():
    from gurobipy import Model, GRB, QuadExpr

    # Create a new model
    m = Model("FastFoodProduction")

    # Decision variables: number of wraps (x) and platters (y)
    x = m.addVar(vtype=GRB.CONTINUOUS, name="wraps", lb=0)
    y = m.addVar(vtype=GRB.CONTINUOUS, name="platters", lb=0)

    # Set objective: minimize total production time
    m.setObjective(10 * x + 8 * y, GRB.MINIMIZE)

    # Add resource constraints
    m.addConstr(5 * x + 7 * y >= 3000, name="meat_constraint")
    m.addConstr(3 * x + 5 * y >= 2500, name="rice_constraint")

    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(x - 3 * y >= 0, name="wraps_vs_platters")
    # Non-linear ratio constraint: x / y >= 3  (i.e., x >= 3 * y, but implemented as x * y >= 3 * y * y)
    # This is a quadratic (non-linear) constraint.
    m.addQConstr(x * y >= 3 * y * y, name="wraps_vs_platters_nonlinear")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal total production time and decision variables
        return m.objVal, x.X, y.X
    else:
        # No feasible solution found
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_wraps_and_platters()
    if result is not None:
        min_time, wraps_opt, platters_opt = result
        print(f"Minimum Total Production Time: {min_time}")
        print(f"Optimal wraps: {wraps_opt}")
        print(f"Optimal platters: {platters_opt}")
    else:
        print("No feasible solution found.")