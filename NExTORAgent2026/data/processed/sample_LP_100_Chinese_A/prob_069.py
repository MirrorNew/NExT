def optimize_fishing():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("FishingOptimization_Nonlinear")

    # Decision variables: acres fished with net (x) and line (y)
    # Allow continuous acres (can be fractional)
    x = m.addVar(name="Net_Acres", lb=0)
    y = m.addVar(name="Line_Acres", lb=0)

    # Set the objective: maximize total fish caught
    m.setObjective(8 * x + 5 * y, GRB.MAXIMIZE)

    # Add constraints

    # Lake area constraint: total acres cannot exceed 250
    m.addConstr(x + y <= 250, name="LakeArea")

    # Bait constraint: 4 units per acre with net, 3 units per acre with line
    m.addConstr(4 * x + 3 * y <= 800, name="Bait")

    # ❤ Non-linearity is introduced. ❤
    # Original linear pain constraint (commented out):
    # m.addConstr(2 * x + y <= 350, name="Pain")

    # New non-linear pain: net pain = 2 * (1.3 ** x), line pain = y
    # Total pain cannot exceed 350
    m.addGenConstrExp(x, m.addVar(name="exp_x"), name="Exp_x")  # auxiliary: exp(x)
    # However, the problem description specifies 1.3^x, not e^x.
    # We model 1.3^x = exp( x * ln(1.3) ).
    # So we need an intermediate variable z = ln(1.3) * x, then exp(z).
    import math
    ln_1_3 = math.log(1.3)

    # z = ln(1.3) * x
    z = m.addVar(name="z", lb=-GRB.INFINITY)
    m.addConstr(z == ln_1_3 * x, name="Def_z")

    # t = exp(z) = 1.3^x
    t = m.addVar(name="pow_1_3_x", lb=0)
    m.addGenConstrExp(z, t, name="Exp_z")

    # Total pain: 2 * (1.3 ** x) + y <= 350
    m.addConstr(2 * t + y <= 350, name="Pain_Nonlinear")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the maximum number of fish caught and the solution
        return {
            "objective": m.objVal,
            "Net_Acres": x.X,
            "Line_Acres": y.X
        }
    else:
        # No feasible solution found
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_fishing()
    if result is not None:
        print(f"Maximum Fish Caught: {result['objective']}")
        print(f"Acres with Net: {result['Net_Acres']}")
        print(f"Acres with Line: {result['Line_Acres']}")
    else:
        print("No feasible solution found.")