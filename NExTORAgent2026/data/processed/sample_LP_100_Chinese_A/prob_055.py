def optimize_meat_production():
    from gurobipy import Model, GRB, QuadExpr

    # Create a new model
    m = Model("Meat_Processing_Optimization_With_Nonlinearity")

    # Decision variables: number of batches of hams and pork ribs
    x = m.addVar(name="Hams", lb=0)
    y = m.addVar(name="Pork_Ribs", lb=0)

    # ❤ Non-linearity is introduced. ❤
    # m.setObjective(150 * x + 300 * y, GRB.MAXIMIZE)

    # Nonlinear (quadratic) objective:
    # Total profit = 150*x + 300*y + 0.01 * x * y * (x + y)
    #              = 150*x + 300*y + 0.01*x^2*y + 0.01*x*y^2
    obj = QuadExpr()
    yy= m.addVar(name="Y", lb=0)
    xx = m.addVar(name="X", lb=0)
    m.addGenConstrPow(x,xx,2)
    m.addGenConstrPow(y,yy,2)
    obj += 150 * x
    obj += 300 * y
    obj += 0.0001 * xx * y     # 0.01 * x^2 * y
    obj += 0.0001 * x * yy     # 0.01 * x * y^2


    m.setObjective(obj, GRB.MAXIMIZE)

    # Add constraints
    # Meat slicer constraint
    m.addConstr(4 * x + 2 * y <= 4000, name="Slicer_Time")
    # Meat packer constraint
    m.addConstr(2.5 * x + 3.5 * y <= 4000, name="Packer_Time")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal profit and decision variables
        return m.objVal, x.X, y.X
    else:
        # No feasible solution found
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_meat_production()
    if result is not None:
        max_profit, x_opt, y_opt = result
        print(f"Optimal hams (x): {x_opt}")
        print(f"Optimal pork ribs (y): {y_opt}")
        print(f"Maximum Total Profit (with nonlinearity): ${max_profit}")
    else:
        print("No feasible solution found.")