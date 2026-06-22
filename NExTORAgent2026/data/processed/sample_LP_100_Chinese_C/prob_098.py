def optimize_grape_transport(
    max_small_crates=100,
    max_large_crates=50,
    min_large_crates=10,
    max_total_crates=60,
    small_crate_capacity=200,
    large_crate_capacity=500
):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("GrapeTransport_NonlinearLike")

    # Decision variables
    x = model.addVar(vtype=GRB.INTEGER, name="small_crates")
    y = model.addVar(vtype=GRB.INTEGER, name="large_crates")

    # ❤ Non-linearity is introduced. ❤
    # Original (linear) objective (commented out due to non-linear modification):
    # model.setObjective(
    #     small_crate_capacity * x + large_crate_capacity * y,
    #     GRB.MAXIMIZE
    # )

    # --- Introduce a binary variable to model the non-linear (step) fixed cost ---
    # z = 1  if  y > 30
    # z = 0  if  y <= 30
    z = model.addVar(vtype=GRB.BINARY, name="large_crate_fixed_cost_trigger")

    # Big-M style linking constraints to enforce z based on y
    M = max_large_crates  # 50 in default parameters

    # If z = 0, then y <= 30
    model.addConstr(y <= 30 + M * z, name="link_y_z_upper")
    # If z = 1, then y >= 31
    model.addConstr(y >= 31 - M * (1 - z), name="link_y_z_lower")

    # Objective:
    # Maximize total transported grapes minus the "penalty" of using >30 large crates.
    # Here, we model the non-linearity (step function) via binary variable z:
    #   if y > 30  → z = 1 → pay fixed cost 2000 (in objective, as a penalty)
    #   if y <= 30 → z = 0 → no fixed cost.
    #
    # NOTE: The original problem's goal is to maximize grapes transported, but
    #       we now also incorporate the operational fixed cost as a penalty term.
    fixed_cost = 2000
    model.setObjective(
        small_crate_capacity * x + large_crate_capacity * y - fixed_cost * z,
        GRB.MAXIMIZE
    )

    # Add constraints
    model.addConstr(x >= 3 * y, name="small_crate_pref")
    model.addConstr(x <= max_small_crates, name="max_small_crates")
    model.addConstr(y <= max_large_crates, name="max_large_crates")
    model.addConstr(x + y <= max_total_crates, name="truck_capacity")
    model.addConstr(y >= min_large_crates, name="min_large_crates")
    model.addConstr(x >= 0, name="nonneg_small")
    model.addConstr(y >= 0, name="nonneg_large")

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        # Return detailed results: crates and effective objective
        result = {
            "small_crates": int(x.X),
            "large_crates": int(y.X),
            "trigger_fixed_cost": int(z.X),
            "total_grapes": small_crate_capacity * x.X + large_crate_capacity * y.X,
            "effective_objective_value": model.objVal
        }
        return result
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    res = optimize_grape_transport()
    if res is not None:
        print(f"Optimal small crates: {res['small_crates']}")
        print(f"Optimal large crates: {res['large_crates']}")
        print(f"Fixed cost triggered (1=yes,0=no): {res['trigger_fixed_cost']}")
        print(f"Total grapes transported (ignoring cost): {res['total_grapes']}")
        print(f"Objective value (grapes - 2000*trigger): {res['effective_objective_value']}")
    else:
        print("No feasible solution found.")