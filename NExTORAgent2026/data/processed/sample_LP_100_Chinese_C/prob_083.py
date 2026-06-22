def optimize_candy_mixes(
    regular_candy_available=80,
    sour_candy_available=60,
    profit_regular_mix=3,
    profit_sour_surprise=5,
    extra_packaging_threshold=30,
    extra_packaging_cost=40
):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("Candy_Mix_Optimization_With_Fixed_Cost")

    # Decision variables: amount of each mix to produce
    x = model.addVar(name="Regular_Mix", lb=0)
    y = model.addVar(name="Sour_Surprise_Mix", lb=0)

    # ❤ Non-linearity is introduced. ❤
    # model.setObjective(profit_regular_mix * x + profit_sour_surprise * y, GRB.MAXIMIZE)
    # Introduce a binary variable to model the fixed cost when y > threshold
    z = model.addVar(vtype=GRB.BINARY, name="Extra_Packaging_Line_Used")

    # Big-M parameter to link y and z (upper bound on possible y)
    M = sour_candy_available  # safe upper bound; can be tightened if needed

    # Objective: profit minus fixed cost if the extra line is used
    model.setObjective(
        profit_regular_mix * x
        + profit_sour_surprise * y
        - extra_packaging_cost * z,
        GRB.MAXIMIZE
    )

    # Add constraints
    # Regular candy constraint
    model.addConstr(
        0.8 * x + 0.1 * y <= regular_candy_available,
        name="Regular_Candy_Constraint"
    )
    # Sour candy constraint
    model.addConstr(
        0.2 * x + 0.9 * y <= sour_candy_available,
        name="Sour_Candy_Constraint"
    )

    # ❤ Non-linearity is introduced. ❤
    # The new constraints below introduce a logical, piecewise behavior via a binary variable
    # linking the production level of sour surprise mix (y) to the fixed cost decision (z).

    # If the extra line is NOT used (z = 0), then y cannot exceed the threshold:
    model.addConstr(
        y <= extra_packaging_threshold + M * z,
        name="Threshold_Upper_Link"
    )

    # If the extra line IS used (z = 1), force y to be strictly greater than the threshold.
    # We approximate "strictly greater" via a small epsilon:
    epsilon = 1e-6
    model.addConstr(
        y >= extra_packaging_threshold + epsilon * z,
        name="Threshold_Lower_Link"
    )

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        return {
            "Maximum_Profit": model.objVal,
            "Regular_Mix_kg": x.X,
            "Sour_Surprise_Mix_kg": y.X,
            "Extra_Line_Used": bool(round(z.X))
        }
    else:
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_candy_mixes()
    if result is not None:
        print(f"Maximum Profit: {result['Maximum_Profit']}")
        print(f"Regular Mix (kg): {result['Regular_Mix_kg']}")
        print(f"Sour Surprise Mix (kg): {result['Sour_Surprise_Mix_kg']}")
        print(f"Extra Packaging Line Used: {result['Extra_Line_Used']}")
    else:
        print("No feasible solution found.")