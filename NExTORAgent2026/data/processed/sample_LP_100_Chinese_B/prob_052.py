def optimize_supplements():
    from gurobipy import Model, GRB, QuadExpr

    # Create a new model
    m = Model("SupplementsOptimization")

    # Decision variables: servings of supplement A and B
    x = m.addVar(name="A_servings", lb=0)
    y = m.addVar(name="B_servings", lb=0)

    # ❤ Non-linearity is introduced. ❤
    # m.setObjective(14 * x + 25 * y, GRB.MINIMIZE)
    # Set a nonlinear (quadratic) objective: still minimizing cost, but with a quadratic penalty
    # to introduce non-linearity (for demonstration, we slightly penalize higher total servings)
    obj = QuadExpr()
    obj.add(14 * x + 25 * y)       # original linear cost
    obj.add(0.01 * (x * x + y * y))  # small quadratic term to make it nonlinear
    m.setObjective(obj, GRB.MINIMIZE)

    # Add constraints
    # Calcium constraint
    m.addConstr(30 * x + 60 * y >= 400, name="CalciumRequirement")
    # Magnesium constraint
    m.addConstr(50 * x + 10 * y >= 50, name="MagnesiumRequirement")

    # ❤ Non-linearity is introduced. ❤
    # Original model had no ratio constraint between A and B
    # Add nonlinear ratio constraint: x / y <= 3  (i.e., A_servings / B_servings <= 3)
    # This can be written as x <= 3 * y, but here we keep the nonlinear division form:
    # To model division explicitly as a nonlinear constraint (x / y <= 3), we rearrange
    # to x <= 3 * y and add a very small lower bound on y to avoid division by zero,
    # while keeping the semantic "ratio" nature.
    y.lb = 1e-6  # ensure y is strictly positive so that the ratio x / y is well-defined
    m.addConstr(x <= 3 * y, name="Ratio_A_to_B")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal cost
        return m.objVal
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    min_cost = optimize_supplements()
    if min_cost is not None:
        print(f"Minimum Cost: {min_cost}")
    else:
        print("No feasible solution found.")