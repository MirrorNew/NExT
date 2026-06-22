def optimize_batches(
    min_regular_batches=10,
    max_resource_medicinal=3000,
    max_resource_rehydration=3500,
    prep_cost=5000
):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("Batch_Production_Optimization_With_Nonlinearity")

    # Decision variables
    # x: number of regular batches
    # y: number of premium batches
    x = model.addVar(name="x", vtype=GRB.INTEGER, lb=min_regular_batches)
    y = model.addVar(name="y", vtype=GRB.INTEGER, lb=0)

    # Binary variable to indicate whether premium batches exceed 30
    z = model.addVar(name="z", vtype=GRB.BINARY)

    # ❤ Non-linearity is introduced. ❤
    # Original linear objective (commented out):
    # model.setObjective(50 * x + 30 * y, GRB.MAXIMIZE)

    # New objective with a fixed preparation cost if y > 30.
    # We are still *maximizing* the number of treated people,
    # but we model the non-linear "trigger" of the 5000 cost via binary z.
    #
    # Interpretation: introduce an auxiliary variable for total treated people
    # and subtract a large penalty (modeled via z) to reflect the fixed cost
    # when y > 30. This makes the problem a mixed-integer (nonlinear in logic,
    # linear in formulation) model.
    people = model.addVar(name="people", vtype=GRB.CONTINUOUS)

    # people = 50x + 30y
    model.addConstr(people == 50 * x + 30 * y, "PeopleCount")

    # Use a big-M formulation to link y and z (non-linear trigger y > 30):
    # If z = 0  ->  y <= 30
    # If z = 1  ->  y can exceed 30 (up to resource limits)
    M = 1000  # sufficiently large upper bound for y
    model.addConstr(y <= 30 + M * z, "Premium_Exceed_30")

    # Objective: maximize treated people minus a fixed "cost" when y > 30
    # (modeled as prep_cost * z).
    model.setObjective(people - prep_cost * z, GRB.MAXIMIZE)

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
        # Return the optimal solution details
        return {
            "max_effective_objective_value": model.objVal,
            "regular_batches": int(x.X),
            "premium_batches": int(y.X),
            "people_treated": people.X,
            "prep_cost_incurred": prep_cost if z.X > 0.5 else 0,
            "z": int(z.X),
        }
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_batches()
    if result is not None:
        print(f"Maximum Effective Objective Value (people - cost): {result['max_effective_objective_value']}")
        print(f"Regular Batches (x): {result['regular_batches']}")
        print(f"Premium Batches (y): {result['premium_batches']}")
        print(f"Total People Treated: {result['people_treated']}")
        print(f"Preparation Cost Incurred: {result['prep_cost_incurred']}")
        print(f"Trigger Variable z (1 if y>30 allowed by model): {result['z']}")
    else:
        print("No feasible solution found.")