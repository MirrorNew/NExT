def optimize_transport(fuel_ship=500, fuel_plane=300, containers_required=500, max_plane_trips=10):
    from gurobipy import Model, GRB, quicksum
    import math

    # Create a new model
    m = Model("TransportOptimization")

    # Decision variables
    x_s = m.addVar(vtype=GRB.INTEGER, name="ship_trips", lb=0)
    x_p = m.addVar(vtype=GRB.INTEGER, name="plane_trips", lb=0)

    # Set objective: minimize total fuel consumption
    m.setObjective(fuel_ship * x_s + fuel_plane * x_p, GRB.MINIMIZE)

    # Add constraints
    # Container requirement
    m.addConstr(40 * x_s + 20 * x_p >= containers_required, "containers_min")
    # Plane trips limit
    m.addConstr(x_p <= max_plane_trips, "max_plane_trips")

    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(x_s >= x_p, "ship_at_least_equal_plane")

    # ---- New non-linear ship-share constraint with seasonal fluctuation ----
    # Average effective ship trips over 12 months:
    # effective_ship = (1/12) * sum_{t=1..12} [ x_s * (1 + 0.1 * cos(pi * x_s * t / 24)) ]
    # Require: effective_ship >= 0.5 * (x_s + x_p)

    t_vals = range(1, 13)
    Y = m.addVars(t_vals)
    COSY = m.addVars(t_vals)
    for t in t_vals:
        m.addConstr(Y[t] == math.pi * x_s * t / 24.0)
        m.addGenConstrCos(Y[t],COSY[t])
    # Build the average effective ship-trips expression
    effective_ship = (1.0 / 12.0) * quicksum(
        x_s * (1.0 + 0.1 * COSY[t] ) for t in t_vals
    )

    # Add the (non-linear) constraint: average effective ship trips ≥ 50% of total trips
    m.addConstr(effective_ship >= 0.5 * (x_s + x_p), name="avg_effective_ship_share")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        total_fuel = m.objVal
        ship_trips = int(x_s.X + 0.5)
        plane_trips = int(x_p.X + 0.5)
        return total_fuel, ship_trips, plane_trips
    else:
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_transport()
    if result is not None:
        min_fuel, ship_trips, plane_trips = result
        print(f"Minimum Total Fuel Consumption: {min_fuel}")
        print(f"Planned Ship Trips (x): {ship_trips}")
        print(f"Planned Plane Trips (y): {plane_trips}")
    else:
        print("No feasible solution found.")