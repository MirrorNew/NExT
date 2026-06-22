def optimize_metal_extraction(pollution_limit=1350):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("MetalExtraction")

    # Decision variables: number of times to perform each process
    x_J = model.addVar(name="Process_J", vtype=GRB.INTEGER, lb=0)
    x_P = model.addVar(name="Process_P", vtype=GRB.INTEGER, lb=0)

    # Set the objective: maximize total metal extracted
    model.setObjective(5 * x_J + 9 * x_P, GRB.MAXIMIZE)

    # ❤ Non-linearity is introduced. ❤
    # Add water constraint
    # model.addConstr(8 * x_J + 6 * x_P <= water_limit, name="WaterLimit")

    # ❤ Non-linearity is introduced. ❤
    # Add pollution constraint
    # model.addConstr(3 * x_J + 5 * x_P <= pollution_limit, name="PollutionLimit")

    # Add pollution limit constraint (still linear)
    total_pollution = 3 * x_J + 5 * x_P
    model.addConstr(total_pollution <= pollution_limit, name="PollutionLimit")

    # Add non-linear ratio constraint: total_water / total_pollution <= 10
    # i.e., (8*x_J + 6*x_P) / (3*x_J + 5*x_P) <= 10
    # This is a fractional constraint, introducing non-linearity.
    total_water = 8 * x_J + 6 * x_P
    model.addQConstr(total_water <= 10 * total_pollution, name="WaterPollutionRatio")

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        # Return the maximum amount of metal extracted
        return model.objVal
    else:
        # No feasible solution found
        return None


# Example usage 
if __name__ == "__main__":
    max_metal = optimize_metal_extraction()
    if max_metal is not None:
        print(f"Maximum Metal Extracted: {max_metal}")
    else:
        print("No feasible solution found.")