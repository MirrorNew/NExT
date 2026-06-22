def optimize_croissants(butter_available=600, flour_available=800,
                        time_almond=12, time_pistachio=10,
                        butter_almond=5, flour_almond=8,
                        butter_pistachio=3, flour_pistachio=6):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("Croissant_Production_MinTime")

    # Decision variables: number of croissants
    x = model.addVar(vtype=GRB.INTEGER, name="AlmondCroissants", lb=0)
    y = model.addVar(vtype=GRB.INTEGER, name="PistachioCroissants", lb=0)

    # ❤ Non-linearity is introduced. ❤
    # model.setObjective(time_almond * x + time_pistachio * y, GRB.MINIMIZE)

    # New variables to capture the piecewise time behavior for almond croissants
    # a = almond croissants produced up to the 50th unit
    # b = almond croissants produced beyond the 50th unit
    a = model.addVar(vtype=GRB.INTEGER, name="AlmondUpTo50", lb=0, ub=50)
    b = model.addVar(vtype=GRB.INTEGER, name="AlmondBeyond50", lb=0)

    # Link almond croissant total to the two parts: x = a + b
    model.addConstr(x == a + b, name="AlmondSplit")

    # Set objective: minimize total production time with step-wise processing time for almond croissants
    # - First 50 almond: 12 minutes each (captured by 'a')
    # - From 51st onward: 14 minutes each (captured by 'b')
    # - Pistachio: always 10 minutes each
    model.setObjective(12 * a + 14 * b + time_pistachio * y, GRB.MINIMIZE)

    # Add resource constraints
    model.addConstr(butter_almond * x + butter_pistachio * y <= butter_available, "ButterLimit")
    model.addConstr(flour_almond * x + flour_pistachio * y <= flour_available, "FlourLimit")

    # Add popularity constraint
    model.addConstr(x >= 3 * y, "PopularityConstraint")

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        # You can also return x, y, etc., but per original code we return the minimized time
        return model.objVal
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    min_time = optimize_croissants()
    if min_time is not None:
        print(f"Minimum Total Production Time: {min_time}")
    else:
        print("No feasible solution found.")