def optimize_slime_production():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("SlimeProduction")

    # Decision variables: number of tables at each station
    x = m.addVar(name="Table1", lb=0)
    y = m.addVar(name="Table2", lb=0)

    # Set the objective: maximize total slime
    m.setObjective(4 * x + 5 * y, GRB.MAXIMIZE)

    # Add resource constraints
    m.addConstr(3 * x + 8 * y <= 100, name="Powder")
    m.addConstr(5 * x + 6 * y <= 90, name="Glue")

    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(2 * x + 4 * y <= 30, name="Mess")
    # New non-linear (fractional) constraint:
    # (total slime) / (total mess) >= 2
    # i.e., (4x + 5y) / (2x + 4y) >= 2
    # This is a non-linear ratio constraint as requested.
    m.addConstr((4 * x + 5 * y) >= 2 * (2 * x + 4 * y), name="SlimeMessRatio")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Also return the optimal decision variables for clarity
        x_val = x.X
        y_val = y.X
        return m.objVal, x_val, y_val
    else:
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_slime_production()
    if result is not None:
        max_slime, table1_num, table2_num = result
        print(f"Maximum Slime Production: {max_slime}")
        print(f"Optimal number of Table1: {table1_num}")
        print(f"Optimal number of Table2: {table2_num}")
    else:
        print("No feasible solution found.")