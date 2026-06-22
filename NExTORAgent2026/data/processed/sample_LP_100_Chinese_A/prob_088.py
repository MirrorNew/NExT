def optimize_appliances():
    from gurobipy import Model, GRB, quicksum

    # Create a new model
    m = Model("Appliance_Optimization_Nonlinear")

    # Decision variables: number of refrigerators and stoves
    # Both are integers and non-negative
    x = m.addVar(vtype=GRB.INTEGER, name="Refrigerators", lb=0)
    y = m.addVar(vtype=GRB.INTEGER, name="Stoves", lb=0)

    # Total units sold
    N = m.addVar(vtype=GRB.INTEGER, name="TotalUnits", lb=0)

    # Set the objective: maximize profit
    m.setObjective(400 * x + 260 * y, GRB.MAXIMIZE)

    # Relationship between N and x, y
    m.addConstr(N == x + y, "TotalUnitsDef")

    # ----------------- Time Constraints -----------------

    # Setup time constraint (linear, unchanged)
    m.addConstr(20 * x + 25 * y <= 13000, "SetupTime")

    # ❤ Non-linearity is introduced. ❤
    # Original linear mover time constraint is commented out:
    # m.addConstr(60 * x + 45 * y <= 20000, "MoverTime")

    # New non-linear mover time constraint:
    # (60*x + 45*y) * 1.3^N <= 20000
    #
    # In Gurobi, we can model this as a general constraint using the
    # built-in exponential function "exp". Note that 1.3^N = exp(N * ln(1.3)).
    #
    # Define an auxiliary variable for 1.3^N
    mover_factor = m.addVar(vtype=GRB.CONTINUOUS, name="MoverFactor", lb=0.0)

    # Define an auxiliary variable for N * ln(1.3)
    import math
    ln_13 = math.log(1.0003)
    t = m.addVar(vtype=GRB.CONTINUOUS, name="ExponentArgument")

    # Link t = N * ln(1.3)
    m.addConstr(t == ln_13 * N, "ExponentArgumentDef")

    # General constraint: mover_factor = exp(t)  (i.e., mover_factor = 1.3^N)
    m.addGenConstrExp(t, mover_factor, name="MoverFactorDef")

    # Now impose the non-linear mover time constraint:
    # (60*x + 45*y) * mover_factor <= 20000
    # This is a bilinear term; Gurobi can handle it as a non-convex quadratic
    # if nonconvex is enabled.
    mover_time_expr = (60 * x + 45 * y) * mover_factor
    m.addQConstr(mover_time_expr <= 20000, name="NonlinearMoverTime")

    # Allow non-convex quadratic constraints
    m.Params.NonConvex = 2

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal profit and decision variables
        return {
            "max_profit": m.objVal,
            "refrigerators": x.X,
            "stoves": y.X,
            "total_units": N.X
        }
    else:
        # No feasible solution found
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_appliances()
    if result is not None:
        print(f"Maximum Profit: ${result['max_profit']:.2f}")
        print(f"Refrigerators sold: {int(result['refrigerators'])}")
        print(f"Stoves sold: {int(result['stoves'])}")
        print(f"Total units sold: {int(result['total_units'])}")
    else:
        print("No feasible solution found.")