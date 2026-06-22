def optimize_slime_production():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("SlimeProduction_NonlinearMess")

    # Decision variables: number of tables at each station
    x = m.addVar(name="Table1", lb=0)
    y = m.addVar(name="Table2", lb=0)

    # ❤ Non-linearity is introduced. ❤
    # Original linear objective (commented out and replaced below):
    # m.setObjective(4 * x + 5 * y, GRB.MAXIMIZE)

    # ❤ Non-linearity is introduced. ❤
    # Introduce binary variable to model the extra 5 units of mess if total tables > 10
    z = m.addVar(vtype=GRB.BINARY, name="ExtraMessTrigger")

    # Set the objective: maximize total slime (unchanged from original problem)
    m.setObjective(4 * x + 5 * y, GRB.MAXIMIZE)

    # Add resource constraints
    m.addConstr(3 * x + 8 * y <= 100, name="Powder")
    m.addConstr(5 * x + 6 * y <= 90, name="Glue")

    # ❤ Non-linearity is introduced. ❤
    # Original mess constraint (without threshold behavior) is commented out:
    # m.addConstr(2 * x + 4 * y <= 30, name="Mess")

    # ❤ Non-linearity is introduced. ❤
    # New mess constraint with step increase of 5 units when total tables > 10
    # If x + y <= 10  →  z = 0  →  2x + 4y <= 30
    # If x + y > 10   →  z = 1  →  2x + 4y + 5 <= 30  →  2x + 4y <= 25
    m.addConstr(2 * x + 4 * y + 5 * z <= 30, name="Mess_with_step")

    # ❤ Non-linearity is introduced. ❤
    # Linking total number of tables with the binary variable z
    # Big-M type constraints to approximate:
    #   z = 0  when x + y <= 10
    #   z = 1  when x + y > 10
    #
    # Choose a sufficiently large M to cover the possible range of x + y.
    M = 1000

    # When z = 0 → x + y <= 10 (tight bound); when z = 1 → x + y can be larger
    m.addConstr(x + y <= 10 + M * z, name="Tables_upper_link")

    # When z = 1 → enforce x + y >= 10 + ε; here we approximate ε = 1
    # So if z = 1, x + y >= 11, modeling "exceeds 10 tables"
    m.addConstr(x + y >= 11 * z, name="Tables_lower_link")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Optionally, you could also return x.X, y.X, z.X if needed
        return m.objVal
    else:
        return None


# Example usage
if __name__ == "__main__":
    max_slime = optimize_slime_production()
    if max_slime is not None:
        print(f"Maximum Slime Production: {max_slime}")
    else:
        print("No feasible solution found.")