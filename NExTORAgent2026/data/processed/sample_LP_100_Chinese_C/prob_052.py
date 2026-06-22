def optimize_supplements():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("SupplementsOptimization")

    # Decision variables: servings of supplement A and B
    x = m.addVar(name="A_servings", lb=0)
    y = m.addVar(name="B_servings", lb=0)

    # ❤ Non-linearity is introduced. ❤
    # Original linear objective (commented out because of the new non-linear / piecewise logic):
    # m.setObjective(14 * x + 25 * y, GRB.MINIMIZE)

    # Introduce a binary variable to model the extra service fee when B_servings > 5
    z = m.addVar(vtype=GRB.BINARY, name="ServiceFeeIndicator")

    # Big-M value: an upper bound on possible servings of B (can be adjusted if a tighter bound is known)
    M = 1000

    # Link binary variable z with condition "y > 5":
    # If y > 5 then z = 1 (extra fee applies); if y <= 5 then z = 0 (no extra fee).
    # These linear constraints enforce that logic:
    m.addConstr(y <= 5 + M * z, name="UpperLink_y_z")
    m.addConstr(y >= 5 + 1e-6 - M * (1 - z), name="LowerLink_y_z")

    # Set the new objective: include the possible one-time service fee of $40
    # When z = 1 (y > 5), an additional 40 is paid; when z = 0 (y <= 5), no extra fee.
    m.setObjective(14 * x + 25 * y + 40 * z, GRB.MINIMIZE)

    # Add constraints
    # Calcium constraint
    m.addConstr(30 * x + 60 * y >= 400, name="CalciumRequirement")
    # Magnesium constraint
    m.addConstr(50 * x + 10 * y >= 50, name="MagnesiumRequirement")

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