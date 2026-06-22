def optimize_vans():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("Vans_Optimization")

    # Decision variables: number of regular and hybrid vans
    x = m.addVar(vtype=GRB.INTEGER, name="RegularVans", lb=0)
    y = m.addVar(vtype=GRB.INTEGER, name="HybridVans", lb=0)

    # ❤ Non-linearity is introduced. ❤
    # Original (now commented out) objective:
    # m.setObjective(x + y, GRB.MINIMIZE)

    # Introduce a binary variable to model the nonlinear “if total vans > 40” condition
    z = m.addVar(vtype=GRB.BINARY, name="Over40Indicator")

    # We use a big-M formulation to link z with the condition x + y > 40.
    # Let M be a sufficiently large upper bound on x + y. A safe and simple choice here:
    M = 100  # upper bound on total vans (can be adjusted if needed)

    # If z = 0, force x + y ≤ 40.
    m.addConstr(x + y <= 40 + M * z, name="TotalVansUpperIfNotOver40")
    # If z = 1, allow x + y to be > 40 by enforcing x + y ≥ 41 (when z=1).
    m.addConstr(x + y >= 41 * z, name="TotalVansLowerIfOver40")

    # Set the objective: still minimize total vans
    m.setObjective(x + y, GRB.MINIMIZE)

    # Add delivery constraint
    # 500 * x + 300 * y >= 20000
    m.addConstr(500 * x + 300 * y >= 20000, name="DeliveryRequirement")

    # ❤ Non-linearity is introduced. ❤
    # Original (now commented out) pollution constraint:
    # m.addConstr(200 * x + 100 * y <= 7000, name="PollutionLimit")

    # New pollution constraint with additional 1000 units if total vans > 40.
    # 200 * x + 100 * y + 1000 * z <= 7000
    m.addConstr(200 * x + 100 * y + 1000 * z <= 7000, name="PollutionLimitWithExtraCost")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal total number of vans and composition
        total_vans = int(round(m.objVal))
        regular_vans = int(round(x.X))
        hybrid_vans = int(round(y.X))
        return total_vans, regular_vans, hybrid_vans
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_vans()
    if result is not None:
        total_vans, regular_vans, hybrid_vans = result
        print(f"Minimum Total Number of Vans: {total_vans}")
        print(f"  Regular Vans: {regular_vans}")
        print(f"  Hybrid Vans:  {hybrid_vans}")
    else:
        print("No feasible solution found.")