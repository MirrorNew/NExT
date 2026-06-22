def optimize_stamping_machines():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("StampingMachines")

    # Decision variables: number of dual and single machines
    x = m.addVar(name="dual_machines", vtype=GRB.INTEGER, lb=0)
    y = m.addVar(name="single_machines", vtype=GRB.INTEGER, lb=0)

    # ❤ Non-linearity is introduced. ❤
    # Binary variable to indicate whether the total number of machines exceeds 10
    z = m.addVar(name="over_10_machines", vtype=GRB.BINARY)

    # Set the objective: minimize total number of machines
    m.setObjective(x + y, GRB.MINIMIZE)

    # Add capacity constraint
    m.addConstr(50 * x + 30 * y >= 300, name="capacity_constraint")

    # ❤ Non-linearity is introduced. ❤
    # Original glue constraint (commented out because non-linear rule is added)
    # m.addConstr(20 * x + 15 * y <= 135, name="glue_constraint")

    # ❤ Non-linearity is introduced. ❤
    # Non-linear glue usage rule linearized with a binary variable:
    # If x + y > 10, then an extra 25 units of glue are used.
    # Glue constraint:
    #   20x + 15y + 25*z <= 135
    m.addConstr(20 * x + 15 * y + 25 * z <= 135, name="glue_constraint_with_extra")

    # To correctly link z with the condition "x + y > 10", use big-M constraints.
    # When z = 0  -> x + y <= 10
    # When z = 1  -> x + y >= 11
    # Choose a sufficiently large M (upper bound on possible x + y). Here M = 100.
    M = 100

    # If z = 0, this enforces x + y <= 10.
    # If z = 1, the right-hand side becomes 10 + M, making the constraint non‑binding.
    m.addConstr(x + y <= 10 + M * z, name="link_upper_total_machines")

    # If z = 1, this enforces x + y >= 11.
    # If z = 0, the right-hand side becomes 11 - M, which is very negative and thus non‑binding.
    m.addConstr(x + y >= 11 - M * (1 - z), name="link_lower_total_machines")

    # Add preference constraint: y > x
    # Since Gurobi does not support strict inequalities directly,
    # we model y >= x + 1 to enforce y > x for integer variables
    m.addConstr(y >= x + 1, name="preference_constraint")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal solution details
        optimal_dual = int(x.X)
        optimal_single = int(y.X)
        optimal_total = int(x.X + y.X)
        return optimal_dual, optimal_single, optimal_total
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_stamping_machines()
    if result is not None:
        dual_machines, single_machines, total_machines = result
        print(f"Optimal number of dual-model machines   : {dual_machines}")
        print(f"Optimal number of single-model machines : {single_machines}")
        print(f"Minimum total number of stamping machines: {total_machines}")
    else:
        print("No feasible solution found.")