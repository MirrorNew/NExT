def optimize_batches(
    min_regular_batches=10,
    max_resource_medicinal=3000,
    max_resource_rehydration=3500
):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("Batch_Production_Optimization")

    # Decision variables
    # x: number of regular batches
    # y: number of premium batches
    x = model.addVar(name="x", vtype=GRB.INTEGER, lb=min_regular_batches)
    y = model.addVar(name="y", vtype=GRB.INTEGER, lb=0)

    # Set objective: maximize total number of treated people
    model.setObjective(50 * x + 30 * y, GRB.MAXIMIZE)

    # Add resource constraints
    model.addConstr(50 * x + 40 * y <= max_resource_medicinal, "Medicinal")
    model.addConstr(40 * x + 60 * y <= max_resource_rehydration, "Rehydration")

    # ❤ Non-linearity is introduced. ❤
    # model.addConstr(x <= y - 1, "BatchRelation")

    # ❤ Non-linearity is introduced. ❤
    # New constraint: x / y <= 0.8  (ratio of regular to premium batches ≤ 0.8)
    # Implemented as a quadratic (non-linear) constraint: x <= 0.8 * y, with y also constrained to be ≥ 1
    # to avoid division by zero in the underlying ratio interpretation.
    model.addQConstr(x <= 0.8 * y, "RatioConstraint")
    model.addConstr(y >= 1, "MinPremiumForRatio")

    # Optional: x ≥ 10 is already enforced by lb=min_regular_batches

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        # Return the maximum total treated people and batch numbers
        return {
            "max_treated_people": model.objVal,
            "regular_batches": int(x.X),
            "premium_batches": int(y.X),
        }
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_batches()
    if result is not None:
        print(f"Maximum Total Number of Treated People: {result['max_treated_people']}")
        print(f"Regular Batches (x): {result['regular_batches']}")
        print(f"Premium Batches (y): {result['premium_batches']}")
    else:
        print("No feasible solution found.")