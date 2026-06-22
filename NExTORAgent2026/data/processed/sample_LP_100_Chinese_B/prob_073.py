def optimize_meal_plan():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("MealOptimization")

    # Decision variables: number of fish and chicken meals
    x = m.addVar(name="fish_meals", lb=0, vtype=GRB.CONTINUOUS)
    y = m.addVar(name="chicken_meals", lb=0, vtype=GRB.CONTINUOUS)

    # Set the objective: minimize total fat intake
    m.setObjective(7 * x + 10 * y, GRB.MINIMIZE)

    # Add constraints
    # Protein constraint
    m.addConstr(10 * x + 15 * y >= 130, name="protein_req")
    # Iron constraint
    m.addConstr(12 * x + 8 * y >= 120, name="iron_req")

    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(y >= 2 * x, name="chicken_pref")
    # Non-linear preference constraint: y / x >= 2  (i.e., y >= 2 * x, but written as a ratio)
    # To introduce non-linearity explicitly, we express the ratio constraint directly:
    # Note: This creates a non-convex quadratic constraint (x * 2 <= y * 1 is linear,
    # but using division y / x >= 2 is non-linear). We model it with a product term.
    # Introduce an auxiliary variable for the ratio and enforce ratio >= 2.
    ratio = m.addVar(name="chicken_to_fish_ratio", lb=0, vtype=GRB.CONTINUOUS)
    # x must be strictly positive for the ratio to be meaningful, so enforce a small lower bound
    m.addConstr(x >= 1e-3, name="positive_fish")

    # Link ratio with x and y: ratio = y / x  -> ratio * x = y  (non-linear term ratio * x)
    m.addQConstr(ratio * x == y, name="ratio_def")

    # Enforce the preference on the ratio directly: y/x >= 2  -> ratio >= 2
    m.addConstr(ratio >= 2, name="chicken_pref_ratio")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal total fat intake and meal plan
        return {
            "total_fat": m.objVal,
            "fish_meals": x.X,
            "chicken_meals": y.X,
            "ratio": ratio.X
        }
    else:
        # No feasible solution
        return None


if __name__ == '__main__':
    result = optimize_meal_plan()
    if result is not None:
        print("Optimal total fat intake:", result["total_fat"])
        print("Fish meals:", result["fish_meals"])
        print("Chicken meals:", result["chicken_meals"])
        print("Chicken/Fish ratio:", result["ratio"])
    else:
        print("No feasible solution found.")