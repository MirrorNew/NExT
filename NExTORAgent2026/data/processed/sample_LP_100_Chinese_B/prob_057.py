def optimize_bakery_profit(oven_hours=70, pastry_hours=32,
                           profit_bagels=20, profit_croissants=40,
                           max_bagels=None, max_croissants=None):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("BakeryOptimization_NonlinearRatio")

    # Decision variables: number of batches of bagels and croissants
    x = model.addVar(name="Bagels")       # batches of bagels
    y = model.addVar(name="Croissants")  # batches of croissants

    # Set the objective: maximize profit
    model.setObjective(profit_bagels * x + profit_croissants * y, GRB.MAXIMIZE)

    # ❤ Non-linearity is introduced. ❤
    # model.addConstr(2 * x + y <= oven_hours, "OvenTime")
    # Use all 70 hours of oven time
    model.addConstr(2 * x + y == oven_hours, "OvenTime_Equality")

    # Add pastry chef time constraint
    model.addConstr(0.25 * x + 2 * y <= pastry_hours, "PastryTime")

    # ❤ Non-linearity is introduced. ❤
    # The ratio of total oven time for bagels to that for croissants
    # should not exceed 4: (2x) / y <= 4  →  2x <= 4y  →  x <= 2y
    # We add this constraint explicitly.
    model.addConstr(x <= 2 * y, "OvenTimeRatio_Bagels_to_Croissants")

    # Optional: add upper bounds if specified
    if max_bagels is not None:
        model.addConstr(x <= max_bagels, "MaxBagels")
    if max_croissants is not None:
        model.addConstr(y <= max_croissants, "MaxCroissants")

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        # Return the maximum profit and optimal production plan
        return {
            "max_profit": model.objVal,
            "bagels": x.X,
            "croissants": y.X
        }
    else:
        # No feasible solution found
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_bakery_profit()
    if result is not None:
        print(f"Maximum profit from bakery operations: ${result['max_profit']:.2f}")
        print(f"Optimal number of bagel batches: {result['bagels']:.2f}")
        print(f"Optimal number of croissant batches: {result['croissants']:.2f}")
    else:
        print("No feasible solution found.")