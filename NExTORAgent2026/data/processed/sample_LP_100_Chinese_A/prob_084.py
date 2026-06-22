def optimize_snacks():
    from gurobipy import Model, GRB, quicksum

    # Create a new model
    m = Model("Snack_Suitcases_Maximize_Nonlinear")

    # Decision variables
    # x: number of small suitcases
    # y: number of large suitcases
    x = m.addVar(vtype=GRB.INTEGER, name="small_suitcases")
    y = m.addVar(vtype=GRB.INTEGER, name="large_suitcases")

    # ❤ Non-linearity is introduced. ❤
    # m.setObjective(50 * x + 80 * y, GRB.MAXIMIZE)

    # Introduce non-linear objective:
    # small suitcases remain linear: 50 * x
    # large suitcases total snacks follow (80 * y)^1.2
    Y = m.addVar()
    m.addConstr(Y == 80 * y)
    YY = m.addVar()
    m.addGenConstrPow(Y,YY,1.02)
    m.setObjective(50 * x +YY, GRB.MAXIMIZE)

    # Add constraints
    m.addConstr(x - 2 * y >= 0, "preference_constraint")
    m.addConstr(x <= 70, "max_small")
    m.addConstr(y <= 50, "max_large")
    m.addConstr(y >= 15, "min_large")
    m.addConstr(x + y <= 70, "total_limit")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the maximum total snacks and decision variables
        return m.objVal, x.X, y.X
    else:
        # No feasible solution
        return None, None, None


# Example usage
if __name__ == "__main__":
    max_snacks, small_cnt, large_cnt = optimize_snacks()
    if max_snacks is not None:
        print(f"Maximum Snacks in Suitcases: {max_snacks}")
        print(f"Small suitcases used: {small_cnt}")
        print(f"Large suitcases used: {large_cnt}")
    else:
        print("No feasible solution found.")