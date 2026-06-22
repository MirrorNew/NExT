import math


def optimize_fertilizer_cost():
    from gurobipy import Model, GRB

    # Create a new model
    # Enable the nonlinear (general) constraint handler
    m = Model("Fertilizer_Optimization")
    m.Params.NonConvex = 2  # allow non-convex nonlinear constraints

    # Decision variables: amount of fertilizers C and Y
    x = m.addVar(name="C", lb=0)  # fertilizer C in kg
    y = m.addVar(name="Y", lb=0)  # fertilizer Y in kg

    # Set the objective: minimize total cost
    m.setObjective(2 * x + 3 * y, GRB.MINIMIZE)

    # Add constraints

    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(1.5 * x + 5 * y >= 5, name="NitrousOxide")
    # Nonlinear nitrous oxide constraint with exponential effect for fertilizer Y:
    # 1.5 * x + 5 * (1.3 ** y) >= 5

    lnpow_1_3_y = m.addVar(name="exp_y")
    pow_1_3_y = m.addVar(name="pow_1_3_y")
    m.addGenConstrLog(
        pow_1_3_y, lnpow_1_3_y        # name for generated exp(y) variable and constraint
    )
    # 假设pow_1_3_y == (1.3 ** y)
    m.addConstr(lnpow_1_3_y == y * math.log(1.5))


    # Now impose the nitrous oxide requirement using 5 * 1.3^y
    m.addConstr(1.5 * x + 5 * pow_1_3_y >= 5, name="NitrousOxide_NL")

    # Vitamin mix constraint remains linear
    m.addConstr(3 * x + y >= 8, name="VitaminMix")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        return m.objVal
    else:
        return None


# Example usage
if __name__ == "__main__":
    min_cost = optimize_fertilizer_cost()
    if min_cost is not None:
        print(f"Minimum Cost of Fertilizer Mixture: {min_cost}")
    else:
        print("No feasible solution found.")