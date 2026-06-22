def optimize_transportation():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("TransportationOptimization")

    # Decision variables
    b = m.addVar(vtype=GRB.INTEGER, name="bikes", lb=0)
    c = m.addVar(vtype=GRB.INTEGER, name="cars", lb=0)

    # ❤ Non-linearity is introduced. ❤
    # Set objective: minimize number of bikes
    # m.setObjective(b, GRB.MINIMIZE)
    # ---- New non-linear(ish) objective with piecewise term for car count > 50 ----
    # Effective bikes = real bikes + 2 * max(c - 50, 0)
    # This is modeled via an auxiliary variable and max() operator.
    extra_parking_cost = m.addVar(vtype=GRB.INTEGER, name="extra_parking_cost", lb=0)

    # extra_parking_cost = max(c - 50, 0)
    m.addConstr(extra_parking_cost >= c - 50, name="extra_parking_lb")
    m.addConstr(extra_parking_cost >= 0, name="extra_parking_nonneg")

    # Objective: minimize equivalent bike count = b + 2 * extra_parking_cost
    m.setObjective(b + 2 * extra_parking_cost, GRB.MINIMIZE)

    # Add capacity constraint
    m.addConstr(3 * b + 5 * c >= 500, name="capacity_constraint")

    # Add vehicle ratio constraint
    # Original: c <= (2/3) * b  corresponding to "cars at most 40% of all vehicles"
    # Explanation: c <= 0.4 * (b + c)  ->  0.6c <= 0.4b  ->  c <= (2/3)b
    m.addConstr(c <= (2.0 / 3.0) * b, name="vehicle_ratio_constraint")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return minimal equivalent bikes, and actual vehicle counts
        return {
            "bikes": int(b.X),
            "cars": int(c.X),
            "extra_parking_cost": int(extra_parking_cost.X),
            "equivalent_bikes": int(b.X + 2 * extra_parking_cost.X),
        }
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_transportation()
    if result is not None:
        print(f"Number of bikes: {result['bikes']}")
        print(f"Number of cars: {result['cars']}")
        print(f"Extra parking cost (in bike-equivalents /2): {result['extra_parking_cost']}")
        print(f"Equivalent total bikes (objective value): {result['equivalent_bikes']}")
    else:
        print("No feasible solution found.")