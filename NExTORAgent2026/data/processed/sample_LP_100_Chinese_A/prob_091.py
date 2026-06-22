def optimize_slime_production():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("SlimeProduction")

    # Decision variables: number of tables at each station
    x = m.addVar(name="Table1", lb=0, vtype=GRB.INTEGER)
    y = m.addVar(name="Table2", lb=0, vtype=GRB.INTEGER)

    # Set the objective: maximize total slime
    m.setObjective(4 * x + 5 * y, GRB.MAXIMIZE)

    # Add resource constraints
    m.addConstr(3 * x + 8 * y <= 100, name="Powder")
    m.addConstr(5 * x + 6 * y <= 90, name="Glue")

    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(2 * x + 4 * y <= 30, name="Mess")
    L = 2 * x + 4 * y
    Y = m.addVar(name="Table3", lb=0)
    m.addConstr(Y == L)
    YY = m.addVar(name="Table4", lb=0)
    m.addGenConstrPow(Y,YY,1.2)
    m.addConstr(YY <= 30 , name="EffectiveMess")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        return m.objVal, x.X, y.X
    else:
        return None, None, None


# Example usage
if __name__ == "__main__":
    max_slime, table1_num, table2_num = optimize_slime_production()
    if max_slime is not None:
        print(f"Maximum Slime Production: {max_slime}")
        print(f"Optimal number of Table1: {table1_num}")
        print(f"Optimal number of Table2: {table2_num}")
    else:
        print("No feasible solution found.")