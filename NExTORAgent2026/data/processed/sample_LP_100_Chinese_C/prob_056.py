def optimize_minty_foam():
    from gurobipy import Model, GRB

    # Create a new model
    model = Model("MintyFoamOptimization")

    # Decision variables: number of demonstrations
    x1 = model.addVar(vtype=GRB.INTEGER, name="Demo1")
    x2 = model.addVar(vtype=GRB.INTEGER, name="Demo2")

    # ❤ Non-linearity is introduced. ❤
    # Original (commented out) linear active ingredient constraint:
    # model.addConstr(20 * x1 + 15 * x2 <= 100, "ActiveIngredientConstraint")

    # Introduce a binary variable to model the step-wise extra 20 units of active ingredient
    # y = 1 if x1 >= 7 (i.e., experiment 1 is run 7 or more times), else y = 0
    y = model.addVar(vtype=GRB.BINARY, name="ExtraSafetyFlag")

    # Big-M value large enough to cover the feasible range of x1
    M = 100

    # Link y with x1: if x1 >= 7 then y must be 1, if x1 <= 6 then y can be 0
    model.addConstr(x1 - 7 * y <= 0, "LinkUpper_x1_y")
    model.addConstr(x1 - 7 + 1 <= M * y, "LinkLower_x1_y")
    # Explanation:
    # 1) x1 - 7*y <= 0:
    #    - if y = 0 → x1 <= 0 (together with integrality and optimality this keeps x1 small unless needed)
    #    - if y = 1 → x1 <= 7 (upper bound near the threshold)
    # 2) x1 - 6 <= M*y:
    #    - if y = 0 → x1 <= 6
    #    - if y = 1 → x1 can be larger (up to M + 6 in theory)

    # Updated active ingredient constraint, including extra 20 units if y=1
    model.addConstr(20 * x1 + 15 * x2 + 20 * y <= 100, "ActiveIngredientConstraint_NonlinearLogic")

    # Set the objective: maximize total minty foam
    model.setObjective(25 * x1 + 18 * x2, GRB.MAXIMIZE)

    # Add resource constraints (unchanged)
    model.addConstr(10 * x1 + 12 * x2 <= 120, "MintConstraint")
    model.addConstr(5 * x1 + 3 * x2 <= 50, "BlackTarConstraint")

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        return {
            "objective_value": model.objVal,
            "Demo1": x1.X,
            "Demo2": x2.X,
            "ExtraSafetyFlag": y.X
        }
    else:
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_minty_foam()
    if result is not None:
        print(f"Maximum Total Minty Foam Produced: {result['objective_value']}")
        print(f"Number of Experiment 1 (Demo1): {result['Demo1']}")
        print(f"Number of Experiment 2 (Demo2): {result['Demo2']}")
        print(f"Extra safety active ingredient triggered (y): {int(result['ExtraSafetyFlag'])}")
    else:
        print("No feasible solution found.")