def optimize_saws():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("woodshop_saws")

    # Decision variables: number of each saw type
    x = m.addVar(vtype=GRB.INTEGER, name="miter_saws", lb=0)
    y = m.addVar(vtype=GRB.INTEGER, name="circular_saws", lb=0)

    # ❤ Non-linearity is introduced. ❤
    # (previous objective without non-linearity)
    # m.setObjective(x + y, GRB.MINIMIZE)

    # Introduce binary variable for the non-linear (piecewise) condition on sawdust
    # z = 1  -> sawdust > 1800, extra dust-removal system installed, max saws = 25
    # z = 0  -> sawdust ≤ 1800, no extra system, max saws = 30
    z = m.addVar(vtype=GRB.BINARY, name="dust_system")

    # Objective: still minimize total number of saws
    m.setObjective(x + y, GRB.MINIMIZE)

    # Add constraints
    # Wood cutting constraint
    m.addConstr(50 * x + 70 * y >= 1500, name="wood_cutting")

    # Sawdust production upper bound (unchanged global limit)
    m.addConstr(60 * x + 100 * y <= 2000, name="sawdust_limit")

    # Link z with sawdust amount using a big-M formulation
    # When z = 0 -> 60x + 100y ≤ 1800
    # When z = 1 -> 60x + 100y can be in (1800, 2000], enforced by the other constraint
    M = 2000  # sufficiently large, consistent with existing global sawdust limit
    m.addConstr(60 * x + 100 * y <= 1800 + M * z,
                name="sawdust_switch_high")
    m.addConstr(60 * x + 100 * y >= 1801 * z,
                name="sawdust_switch_low")

    # Capacity constraint depending on z:
    # If z = 0 (sawdust ≤ 1800): x + y ≤ 30
    # If z = 1 (sawdust > 1800): x + y ≤ 25
    m.addConstr(x + y <= 30 - 5 * z, name="saw_capacity")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal total number of saws and breakdown
        return {
            "total_saws": int(m.objVal),
            "miter_saws": int(x.X),
            "circular_saws": int(y.X),
            "dust_system_installed": int(z.X),
            "sawdust": 60 * x.X + 100 * y.X,
            "boards_cut": 50 * x.X + 70 * y.X,
        }
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_saws()
    if result is not None:
        print(f"Minimum Total Number of Saws: {result['total_saws']}")
        print(f"  Miter saws: {result['miter_saws']}")
        print(f"  Circular saws: {result['circular_saws']}")
        print(f"  Dust system installed (1=yes,0=no): {result['dust_system_installed']}")
        print(f"  Total sawdust: {result['sawdust']}")
        print(f"  Total boards cut: {result['boards_cut']}")
    else:
        print("No feasible solution found.")