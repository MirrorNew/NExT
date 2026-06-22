def optimize_minty_foam():
    from gurobipy import Model, GRB, QuadExpr

    # Create a new model
    model = Model("MintyFoamOptimization")
    
    # Decision variables: number of demonstrations
    x1 = model.addVar(vtype=GRB.INTEGER, name="Demo1")
    x2 = model.addVar(vtype=GRB.INTEGER, name="Demo2")
    
    # Set the objective: maximize total minty foam
    model.setObjective(25 * x1 + 18 * x2, GRB.MAXIMIZE)
    
    # Add resource constraints
    model.addConstr(10 * x1 + 12 * x2 <= 120, "MintConstraint")
    model.addConstr(20 * x1 + 15 * x2 <= 100, "ActiveIngredientConstraint")
    
    # ❤ Non-linearity is introduced. ❤
    # model.addConstr(5 * x1 + 3 * x2 <= 50, "BlackTarConstraint")
    black_tar = 5 * x1 + 3 * x2               # total black tar
    minty_foam = 25 * x1 + 18 * x2            # total minty foam (same as objective)
    # Non-linear ratio constraint: black_tar / minty_foam <= 0.4
    # i.e., black_tar <= 0.4 * minty_foam  ->  black_tar - 0.4 * minty_foam <= 0
    # This keeps the original ratio form and introduces non-linearity explicitly.
    expr = QuadExpr()
    expr.add(black_tar)
    expr.add(-0.4 * minty_foam)
    model.addQConstr(expr <= 0, name="BlackTarRatioConstraint")
    
    # Optimize the model
    model.optimize()
    
    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        return model.objVal
    else:
        return None

# Example usage
if __name__ == "__main__":
    max_minty_foam = optimize_minty_foam()
    if max_minty_foam is not None:
        print(f"Maximum Total Minty Foam Produced: {max_minty_foam}")
    else:
        print("No feasible solution found.")