def optimize_croissants(
    t,
    butter_available=600,
    flour_available=800,
    time_almond=12,
    time_pistachio=10,
    butter_almond=5,
    flour_almond=8,
    butter_pistachio=3,
    flour_pistachio=6,
):
    import gurobipy as gp
    from gurobipy import GRB
    import math

    # Create a new model
    model = gp.Model("Croissant_Production_MinTime_Nonlinear")

    # Decision variables: number of croissants in batch t
    x = model.addVar(vtype=GRB.CONTINUOUS, name="AlmondCroissants", lb=0)
    y = model.addVar(vtype=GRB.CONTINUOUS, name="PistachioCroissants", lb=0)

    # Set objective: minimize total production time of batch t
    model.setObjective(time_almond * x + time_pistachio * y, GRB.MINIMIZE)

    # Add resource constraints
    model.addConstr(
        butter_almond * x + butter_pistachio * y <= butter_available, "ButterLimit"
    )
    model.addConstr(
        flour_almond * x + flour_pistachio * y <= flour_available, "FlourLimit"
    )

    # ❤ Non-linearity is introduced. ❤
    # model.addConstr(x >= 3 * y, "PopularityConstraint")

    # Add non-linear popularity constraint for batch t:
    # x >= 3 * y * [1 + 0.1 * cos(pi * y * t / 50)]

    Y = model.addVar(vtype=GRB.CONTINUOUS)
    XXX = model.addVar(vtype=GRB.CONTINUOUS)
    model.addConstr(XXX == math.pi * y * t / 50.0)
    COSXXX = model.addVar(vtype=GRB.CONTINUOUS)
    model.addGenConstrCos(XXX, COSXXX)
    model.addConstr(Y == (1 + 0.1 * COSXXX))
    expr_nl = 3 * y * Y
    model.addConstr(x >= expr_nl, "PopularityConstraint_Nonlinear")

    # Optimize the model (need nonconvex option due to cos)
    model.Params.NonConvex = 2
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        # Return the decision variables and the total minimized time for this batch t
        return {
            "t": t,
            "AlmondCroissants": x.X,
            "PistachioCroissants": y.X,
            "MinTotalTime": model.objVal,
        }
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    # Solve the problem for a given batch index t (1 to 12)
    batch_t = 5  # example: 5th batch in the month
    result = optimize_croissants(t=batch_t)
    if result is not None:
        print(f"Batch t = {result['t']}")
        print(f"Almond croissants: {result['AlmondCroissants']:.4f}")
        print(f"Pistachio croissants: {result['PistachioCroissants']:.4f}")
        print(f"Minimum Total Production Time: {result['MinTotalTime']:.4f}")
    else:
        print("No feasible solution found for this batch.")