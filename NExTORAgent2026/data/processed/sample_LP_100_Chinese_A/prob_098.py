def optimize_grape_transport(
    t,  # month index: 1..12
    max_small_crates=100,
    max_large_crates=50,
    min_large_crates=10,
    base_max_total_crates=60,
    fluctuation_amplitude=0.1,
    pi_value=3.14,
    small_crate_capacity=200,
    large_crate_capacity=500
):
    import gurobipy as gp
    from gurobipy import GRB
    import math

    # Create a new model
    model = gp.Model("GrapeTransportSeasonal")

    # Decision variables
    x = model.addVar(vtype=GRB.INTEGER, name="small_crates")
    y = model.addVar(vtype=GRB.INTEGER, name="large_crates")

    # Set objective: maximize total grapes
    model.setObjective(
        small_crate_capacity * x + large_crate_capacity * y,
        GRB.MAXIMIZE
    )

    # Add constraints independent of t
    model.addConstr(x >= 3 * y, name="small_crate_pref")
    model.addConstr(x <= max_small_crates, name="max_small_crates")
    model.addConstr(y <= max_large_crates, name="max_large_crates")
    model.addConstr(y >= min_large_crates, name="min_large_crates")
    model.addConstr(x >= 0, name="nonneg_small")
    model.addConstr(y >= 0, name="nonneg_large")

    # ❤ Non-linearity is introduced. ❤
    # model.addConstr(x + y <= max_total_crates, name="truck_capacity")

    # Compute month-dependent (nonlinear) truck capacity:
    # max_total_crates(t) = 60 * [1 + 0.1 * cos(pi * t / 6)]
    seasonal_max_total_crates = base_max_total_crates * (
        1 + fluctuation_amplitude * math.cos(pi_value * t / 6.0)
    )
    model.addConstr(x + y <= seasonal_max_total_crates, name=f"truck_capacity_month_{t}")

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        # Return the optimal numbers and maximum total grapes transported
        return {
            "month": t,
            "small_crates": int(x.X),
            "large_crates": int(y.X),
            "max_grapes": model.objVal,
            "seasonal_capacity": seasonal_max_total_crates
        }
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    # Solve for each month t = 1..12 and print the optimal plan
    results = []
    for t in range(1, 13):
        res = optimize_grape_transport(t=t)
        if res is not None:
            results.append(res)
            print(
                f"Month {res['month']}: "
                f"Seasonal capacity ≈ {res['seasonal_capacity']:.2f} crates, "
                f"small crates = {res['small_crates']}, "
                f"large crates = {res['large_crates']}, "
                f"max grapes = {res['max_grapes']}"
            )
        else:
            print(f"Month {t}: No feasible solution found.")