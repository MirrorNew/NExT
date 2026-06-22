def optimize_supplements():
    from gurobipy import Model, GRB

    # Create a new model
    model = Model("SupplementsOptimization_Nonlinear")

    # Decision variables: servings of supplement A and B, and continuous days T
    x = model.addVar(name="A_servings", lb=0,ub=10)
    y = model.addVar(name="B_servings", lb=0,ub=6)
    T = model.addVar(name="Days_T", lb=7)

    # ❤ Non-linearity is introduced. ❤
    # model.setObjective(14 * x + 25 * y, GRB.MINIMIZE)
    # New nonlinear objective: 14 * A_servings * B_servings * T + 25 * B_servings
    Y = model.addVar(name="Y_servings", lb=0)
    model.addConstr(y * T == Y)
    model.setObjective(14 * x * Y + 25 * y, GRB.MINIMIZE)

    # Add constraints
    # Calcium constraint: 30x + 60y ≥ 400
    model.addConstr(30 * x + 60 * y >= 400, name="CalciumRequirement")
    # Magnesium constraint: 50x + 10y ≥ 50
    model.addConstr(50 * x + 10 * y >= 50, name="MagnesiumRequirement")

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        # Return the optimal solution values and objective
        return {
            "A_servings": x.X,
            "B_servings": y.X,
            "Days_T": T.X,
            "Min_Daily_Cost": model.objVal
        }
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_supplements()
    if result is not None:
        print(f"Optimal A servings per day: {result['A_servings']}")
        print(f"Optimal B servings per day: {result['B_servings']}")
        print(f"Optimal continuous days T: {result['Days_T']}")
        print(f"Minimum nonlinear daily cost: {result['Min_Daily_Cost']}")
    else:
        print("No feasible solution found.")