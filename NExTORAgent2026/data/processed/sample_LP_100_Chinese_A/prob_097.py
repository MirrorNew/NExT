import math


def optimize_bakery_production():
    from gurobipy import Model, GRB, quicksum

    # Create a new model
    m = Model("BakeryProduction")

    # Decision variables: number of brownies and lemon squares
    B = m.addVar(name="Brownies", vtype=GRB.INTEGER, lb=0)
    L = m.addVar(name="LemonSquares", vtype=GRB.INTEGER, lb=0)

    # Set the objective: minimize total fiber
    m.setObjective(4 * B + 6 * L, GRB.MINIMIZE)

    # Add resource constraints
    m.addConstr(5 * B <= 2500, name="ChocolateConstraint")
    m.addConstr(7 * L <= 3300, name="LemonConstraint")

    # Lemon squares must be more than brownies
    # Note: Gurobi does not support strict inequalities directly.
    # To model L > B, we can use L >= B + 1
    m.addConstr(L >= B + 1, name="L_greater_than_B")

    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(3 * B >= 2 * L, name="BrowniePercentage")

    # ---- Non-linear effective-brownie share constraint ----
    # Effective brownies: Beff = B * 1.3^(B - 1)
    # Requirement:
    #   Beff / (Beff + L) >= 0.4
    # <=> Beff >= 0.4 * (Beff + L)
    # <=> 0.6 * Beff >= 0.4 * L
    # <=> 3 * Beff >= 2 * L
    # with Beff = B * 1.3^(B - 1)
    growth_factor = 1.3
    lnX = m.addVar()
    X = m.addVar()
    Beff = m.addVar()
    m.addConstr(lnX == (B - 1) * math.log(growth_factor))
    m.addGenConstrExp(lnX,X)
    m.addConstr(Beff == B * X)

    m.addConstr(3 * Beff >= 2 * L, name="Nonlinear_BrownieShare")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the minimal fiber usage, and also the optimal B, L
        return m.objVal, B.X, L.X
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_bakery_production()
    if result is not None:
        min_fiber, opt_B, opt_L = result
        print(f"Minimum Total Fiber Usage: {min_fiber}")
        print(f"Optimal Brownies (B): {opt_B}")
        print(f"Optimal Lemon Squares (L): {opt_L}")
    else:
        print("No feasible solution found.")