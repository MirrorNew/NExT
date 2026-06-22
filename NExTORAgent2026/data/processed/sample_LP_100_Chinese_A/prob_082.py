def optimize_water_transport():
    from gurobipy import Model, GRB, quicksum
    import math

    # Create a new model
    m = Model("Water_Transport_Maximize")

    # Decision variables
    # x: number of small kegs
    # y: number of large kegs
    x = m.addVar(vtype=GRB.INTEGER, name="small_kegs")
    y = m.addVar(vtype=GRB.INTEGER, name="large_kegs")

    # Optional: month index t as an integer decision variable (1 to 12)
    # If you instead want to fix a specific month, you can set t_value directly.
    t = m.addVar(vtype=GRB.INTEGER, lb=1, ub=12, name="month")

    # Set objective: maximize total liters of water transported
    m.setObjective(40 * x + 100 * y, GRB.MAXIMIZE)

    # Add constraints
    m.addConstr(x <= 30, "max_small_kegs")
    m.addConstr(y <= 10, "max_large_kegs")

    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(x >= 2 * y, "small_at_least_twice_large")

    # Non-linear seasonal ratio constraint:
    # x ≥ y × 2 × [1 + 0.1 × cos(π·y·t/12)]
    # where π = 3.1416, t ∈ {1,...,12}
    pi_val = 3.1416

    # Gurobi's Python interface allows general constraints via addGenConstr or
    # direct non-linear expressions when using the non-convex solver features.
    # Here we directly build the (non-linear) expression using math.cos on a
    # symbolic argument constructed from variables x, y, t.
    #
    # Note: This requires setting the appropriate parameter for non-convex models
    # (NonConvex = 2) in a real run. That is not shown here, as the focus is on
    # introducing the non-linearity in the model structure.
    #
    # We model:
    # x >= 2 * y * (1 + 0.1 * cos(pi * y * t / 12))
    #
    # Because Gurobi does not natively support arbitrary non-linear functions
    # in its standard MILP interface, this is a conceptual illustration of
    # how the non-linear term would appear in code.
    #
    # If you wanted a fully solvable Gurobi model, you would need to:
    #   - discretize y and t (they are already integers with small domains),
    #   - precompute cos(pi * y * t / 12) values,
    #   - and use additional linear constraints / auxiliary variables
    #     to model this as a piecewise or enumerated non-linear relationship.
    #
    # For the purpose of this task, we introduce the non-linearity directly:

    Y = m.addVar()
    m.addConstr(Y == pi_val * y * t / 12.0)
    COSY = m.addVar()
    m.addGenConstrCos(Y,COSY)
    nonlinear_expr = 2 * y * (1 + 0.1 * COSY)
    m.addConstr(x >= nonlinear_expr, "seasonal_small_vs_large")

    m.addConstr(x + y <= 25, "total_kegs_limit")
    m.addConstr(y >= 5, "min_large_kegs")
    m.addConstr(x >= 0, "non_negative_small")
    m.addConstr(y >= 0, "non_negative_large")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the maximum total water transported
        return m.objVal
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    max_water = optimize_water_transport()
    if max_water is not None:
        print(f"Maximum Total Water Transported: {max_water} liters")
    else:
        print("No feasible solution found.")