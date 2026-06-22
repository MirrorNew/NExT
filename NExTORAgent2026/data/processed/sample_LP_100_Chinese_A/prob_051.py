def optimize_stamping_machines():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("StampingMachines_Nonlinear")

    # Decision variables: number of dual and single machines
    x = m.addVar(name="dual_machines", vtype=GRB.INTEGER, lb=0)
    y = m.addVar(name="single_machines", vtype=GRB.INTEGER, lb=0)

    # Auxiliary variable for nonlinear glue usage of single machines
    g_single = m.addVar(name="glue_single_nonlinear", lb=0)

    # Set the objective: minimize total number of machines
    m.setObjective(x + y, GRB.MINIMIZE)

    # Add capacity constraint
    m.addConstr(50 * x + 30 * y >= 300, name="capacity_constraint")

    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(20 * x + 15 * y <= 135, name="glue_constraint")
    # Nonlinear glue usage: single machines consume (15*y)^1.2 units per minute
    Y = m.addVar()
    m.addConstr(Y == 15 * y)
    m.addGenConstrPow(Y, g_single, 1.002, name="single_glue_power")
    m.addConstr(20 * x + g_single <= 150, name="glue_constraint_nonlinear")

    # Add preference constraint: y > x  (implemented as y >= x + 1)
    m.addConstr(y >= x + 1, name="preference_constraint")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal total number of machines and the breakdown
        return {
            "total_machines": m.objVal,
            "dual_machines": x.X,
            "single_machines": y.X,
            "single_glue_nonlinear": g_single.X,
        }
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_stamping_machines()
    if result is not None:
        print(f"Minimum Total Number of Stamping Machines: {result['total_machines']}")
        print(f"Dual-model machines (x): {result['dual_machines']}")
        print(f"Single-model machines (y): {result['single_machines']}")
        print(
            f"Nonlinear glue usage of single-model machines ( (15*y)^1.2 ): "
            f"{result['single_glue_nonlinear']}"
        )
    else:
        print("No feasible solution found.")