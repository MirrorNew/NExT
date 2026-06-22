def optimize_fishing():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("FishingOptimization")

    # Decision variables: acres fished with net (x) and line (y)
    x = m.addVar(name="Net_Acres", lb=0)
    y = m.addVar(name="Line_Acres", lb=0)

    # ❤ Non-linearity is introduced. ❤
    # m.setObjective(8 * x + 5 * y, GRB.MAXIMIZE)
    # Introduce an auxiliary variable to model piecewise pain for net usage:
    # z = max(0, x - 150), i.e., acres of net usage above 150
    z = m.addVar(name="Net_Above_150", lb=0)

    # Same objective: maximize total fish caught (still linear in x, y)
    m.setObjective(8 * x + 5 * y, GRB.MAXIMIZE)

    # Add constraints
    m.addConstr(x + y <= 250, name="LakeArea")
    m.addConstr(4 * x + 3 * y <= 800, name="Bait")

    # ❤ Non-linearity is introduced. ❤
    # Original (linear) pain constraint:
    # m.addConstr(2 * x + y <= 350, name="Pain")
    # Modified pain constraint with piecewise net pain:
    # For x ≤ 150: pain = 2x + y
    # For x > 150: pain = 2*150 + 3*(x - 150) + y = 3x - 150 + y
    # Implemented via z = max(0, x - 150), so:
    # pain = 2x + z + y  (because z contributes the extra 1 unit per acre above 150)
    m.addConstr(z >= x - 150, name="NetAbove150_lb")
    m.addConstr(z >= 0, name="NetAbove150_nonneg")
    m.addConstr(2 * x + z + y <= 350, name="Pain_Piecewise")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the maximum number of fish caught and the decision variables
        return m.objVal, x.X, y.X
    else:
        # No feasible solution found
        return None, None, None


# Example usage
if __name__ == "__main__":
    max_fish, net_acres, line_acres = optimize_fishing()
    if max_fish is not None:
        print(f"Maximum Fish Caught: {max_fish}")
        print(f"Optimal Net Acres: {net_acres}")
        print(f"Optimal Line Acres: {line_acres}")
    else:
        print("No feasible solution found.")