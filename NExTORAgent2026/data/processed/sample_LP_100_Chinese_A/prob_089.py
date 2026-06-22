def optimize_grilled_cheese():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("GrilledCheeseOptimization")

    # Decision variables: number of light and heavy sandwiches
    x = m.addVar(name="light_sandwiches", vtype=GRB.INTEGER, lb=0)
    y = m.addVar(name="heavy_sandwiches", vtype=GRB.INTEGER, lb=0)

    # ❤ Non-linearity is introduced. ❤
    # m.setObjective(10 * x + 15 * y, GRB.MINIMIZE)
    # Introduce non-linear total time: 10*x + (15*y)^1.2
    Y = m.addVar()
    YY = m.addVar()
    m.addConstr(Y == 15 * y)
    m.addGenConstrPow(Y,YY,1.2)
    m.setObjective(10 * x + YY, GRB.MINIMIZE)

    # Add resource constraints
    m.addConstr(2 * x + 3 * y <= 300, name="bread_constraint")
    m.addConstr(3 * x + 5 * y <= 500, name="cheese_constraint")

    # Add demand ratio constraint
    m.addConstr(y >= 3 * x, name="heavy_light_ratio")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal total production time and decision variables
        return m.objVal, x.X, y.X
    else:
        # No feasible solution found
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_grilled_cheese()
    if result is not None:
        min_time, x_opt, y_opt = result
        print(f"Optimal number of light sandwiches (x): {x_opt}")
        print(f"Optimal number of heavy sandwiches (y): {y_opt}")
        print(f"Minimum Total Production Time: {min_time}")
    else:
        print("No feasible solution found.")