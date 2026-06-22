def optimize_stamping_machines():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("StampingMachines")

    # Decision variables: number of dual and single machines
    x = m.addVar(name="dual_machines", vtype=GRB.INTEGER, lb=0)
    y = m.addVar(name="single_machines", vtype=GRB.INTEGER, lb=0)

    # Set the objective: minimize total number of machines
    m.setObjective(x + y, GRB.MINIMIZE)

    # Add capacity constraint
    m.addConstr(50 * x + 30 * y >= 300, name="capacity_constraint")

    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(20 * x + 15 * y <= 135, name="glue_constraint")
    # Introduce the non-linear glue ratio constraint:
    # single-model total glue / dual-model total glue <= 9/8
    # 15*y / (20*x) <= 9/8  →  15*y <= (9/8)*20*x = (45/2)*x
    # Equivalently: 30*y <= 45*x
    m.addConstr(30 * y <= 45 * x, name="glue_ratio_constraint")

    # Add preference constraint: y > x
    # Since Gurobi does not support strict inequalities directly,
    # we model y >= x + 1 to enforce y > x for integer variables
    m.addConstr(y >= x + 1, name="preference_constraint")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal total number of machines
        return m.objVal
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    min_machines = optimize_stamping_machines()
    if min_machines is not None:
        print(f"Minimum Total Number of Stamping Machines: {min_machines}")
    else:
        print("No feasible solution found.")