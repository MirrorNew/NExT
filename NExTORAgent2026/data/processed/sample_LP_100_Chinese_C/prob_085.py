def optimize_metal_extraction(water_limit=1500, pollution_limit=1350):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("MetalExtraction")

    # Decision variables: number of times to perform each process
    x_J = model.addVar(name="Process_J", vtype=GRB.INTEGER, lb=0)
    x_P = model.addVar(name="Process_P", vtype=GRB.INTEGER, lb=0)

    # ❤ Non-linearity is introduced. ❤
    # Original (commented out) single variable for process P water use:
    # x_P = model.addVar(name="Process_P", vtype=GRB.INTEGER, lb=0)

    # ❤ Non-linearity is introduced. ❤
    # Split process P into two parts:
    # x_P1: times using process P within the first 120 uses (each uses 6 units of water)
    # x_P2: times using process P beyond 120 uses (each uses 8 units of water)
    x_P1 = model.addVar(name="Process_P_1_to_120", vtype=GRB.INTEGER, lb=0, ub=120)
    x_P2 = model.addVar(name="Process_P_over_120", vtype=GRB.INTEGER, lb=0)

    # Relate total P usage x_P to its components x_P1 and x_P2
    model.addConstr(x_P == x_P1 + x_P2, name="P_Split")

    # Set the objective: maximize total metal extracted
    # Metal from J: 5 per use; from P: 9 per use (regardless of segment)
    # So objective uses total x_P, which equals x_P1 + x_P2
    model.setObjective(5 * x_J + 9 * x_P, GRB.MAXIMIZE)

    # ❤ Non-linearity is introduced. ❤
    # Original (commented out) water constraint with uniform water usage for P:
    # model.addConstr(8 * x_J + 6 * x_P <= water_limit, name="WaterLimit")

    # New water constraint reflecting piecewise water usage for P:
    # First 120 uses of P (x_P1): 6 units of water each
    # Beyond 120 uses of P (x_P2): 8 units of water each
    model.addConstr(8 * x_J + 6 * x_P1 + 8 * x_P2 <= water_limit, name="WaterLimit_Piecewise")

    # Add pollution constraint (unchanged: P always generates 5 units pollution per use)
    model.addConstr(3 * x_J + 5 * x_P <= pollution_limit, name="PollutionLimit")

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        # Return the decision variable values and the maximum amount of metal extracted
        return {
            "Process_J": int(x_J.X),
            "Process_P_total": int(x_P.X),
            "Process_P_1_to_120": int(x_P1.X),
            "Process_P_over_120": int(x_P2.X),
            "Max_Metal_Extracted": model.objVal
        }
    else:
        # No feasible solution found
        return None


# Example usage 
if __name__ == "__main__":
    result = optimize_metal_extraction()
    if result is not None:
        print(f"Process J uses: {result['Process_J']}")
        print(f"Process P total uses: {result['Process_P_total']}")
        print(f"  - P within first 120 uses: {result['Process_P_1_to_120']}")
        print(f"  - P beyond 120 uses: {result['Process_P_over_120']}")
        print(f"Maximum Metal Extracted: {result['Max_Metal_Extracted']}")
    else:
        print("No feasible solution found.")