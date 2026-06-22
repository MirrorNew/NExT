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

    # ❤ Non-linearity is introduced. ❤
    # model.setObjective(50 * x + 30 * y, GRB.MAXIMIZE)
    # Set nonlinear objective: total treated people
    # Regular batches: 50 people per batch  →  50 * x
    # Premium batches: (30 * y)^1.2        →  (30 * y) ** 1.2
    Y = model.addVar()
    YY = model.addVar()
    model.addConstr(Y == 30*y)
    model.addGenConstrPow(Y,YY,1.02)
    model.setObjective(50 * x + YY, GRB.MAXIMIZE)

    # Add resource constraints
    model.addConstr(50 * x + 40 * y <= max_resource_medicinal, "Medicinal")
    model.addConstr(40 * x + 60 * y <= max_resource_rehydration, "Rehydration")

    # Add batch relationship constraint: x < y
    # Since variables are integers, x <= y - 1
    model.addConstr(x <= y - 1, "BatchRelation")

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        # Return the optimal numbers of batches and the maximum total treated people
        return {
            "regular_batches": int(x.X),
            "premium_batches": int(y.X),
            "max_treated_people": model.objVal,
        }
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_batches()
    if result is not None:
        print(f"Optimal number of regular batches: {result['regular_batches']}")
        print(f"Optimal number of premium batches: {result['premium_batches']}")
        print(f"Maximum Total Number of Treated People: {result['max_treated_people']}")
    else:
        print("No feasible solution found.")