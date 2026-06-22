def optimize_transport(fuel_ship=500,
                       fuel_plane=300,
                       containers_required=500,
                       max_plane_trips=10):
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("TransportOptimization")

    # Decision variables
    x_s = m.addVar(vtype=GRB.INTEGER, name="ship_trips", lb=0)
    x_p = m.addVar(vtype=GRB.INTEGER, name="plane_trips", lb=0)

    # ❤ Non-linearity is introduced. ❤
    # m.setObjective(fuel_ship * x_s + fuel_plane * x_p, GRB.MINIMIZE)

    # Binary variable for triggering the extra fuel cost when ship trips > 8
    y = m.addVar(vtype=GRB.BINARY, name="extra_cost_indicator")

    # Big-M for linking y with the logical condition (x_s > 8)
    M = 1000  # sufficiently large number (larger than any realistic # of trips)

    # Extra fuel cost (2000 liters) if x_s > 8, else 0
    extra_cost = 2000

    # Set new (mixed-integer) objective: base fuel + possible extra cost
    m.setObjective(fuel_ship * x_s + fuel_plane * x_p + extra_cost * y, GRB.MINIMIZE)

    # Add constraints
    # Container requirement
    m.addConstr(40 * x_s + 20 * x_p >= containers_required, "containers_min")
    # Plane trips limit
    m.addConstr(x_p <= max_plane_trips, "max_plane_trips")
    # Ship trips at least 50% of total trips  (x_s >= x_p)
    m.addConstr(x_s >= x_p, "ship_at_least_equal_plane")

    # Logic: y = 1  ⇔  x_s > 8 (modeled with Big-M inequalities)
    # If y = 0 ⇒ x_s ≤ 8
    m.addConstr(x_s <= 8 + M * y, "ship_trips_upper_if_no_extra")
    # If y = 1 ⇒ x_s ≥ 9
    m.addConstr(x_s >= 9 * y, "ship_trips_lower_if_extra")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        total_fuel = m.objVal
        ship_trips = x_s.X
        plane_trips = x_p.X
        return total_fuel, ship_trips, plane_trips
    else:
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_transport()
    if result is not None:
        min_fuel, ship_trips, plane_trips = result
        print(f"Minimum Total Fuel Consumption (including possible extra cost): {min_fuel}")
        print(f"Optimal ship trips: {ship_trips}")
        print(f"Optimal plane trips: {plane_trips}")
    else:
        print("No feasible solution found.")