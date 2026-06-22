def optimize_meal_delivery():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("MealDeliveryOptimization")

    # Decision variables
    x_b = m.addVar(vtype=GRB.INTEGER, name="bikes")
    x_s = m.addVar(vtype=GRB.INTEGER, name="scooters")

    # ❤ Non-linearity is introduced. ❤
    # Binary variable to indicate whether peak system is activated
    y_peak = m.addVar(vtype=GRB.BINARY, name="peak_system")

    # Set objective: maximize total meals
    m.setObjective(8 * x_b + 5 * x_s, GRB.MAXIMIZE)

    # Add constraints
    # ❤ Non-linearity is introduced. ❤
    # Original charge constraint (commented out):
    # m.addConstr(3 * x_b + 2 * x_s <= 200, name="charge_limit")
    # Modified charge constraint with extra 10 units if total meals >= 250
    m.addConstr(3 * x_b + 2 * x_s + 10 * y_peak <= 200, name="charge_limit_with_peak")

    # ❤ Non-linearity is introduced. ❤
    # Link peak system activation with total meals using big-M formulation
    # If y_peak = 0  =>  8*x_b + 5*x_s <= 249  (no peak system)
    # If y_peak = 1  =>  8*x_b + 5*x_s >= 250 (peak system required)
    # Upper bound side: total meals <= 249 + M * y_peak
    M = 1000
    m.addConstr(8 * x_b + 5 * x_s <= 249 + M * y_peak, name="peak_upper_link")
    # Lower bound side: total meals >= 250 * y_peak
    m.addConstr(8 * x_b + 5 * x_s >= 250 * y_peak, name="peak_lower_link")

    # Bike usage limit (30% of total vehicles)
    m.addConstr(7 * x_b <= 3 * x_s, name="bike_ratio")
    # Minimum scooters
    m.addConstr(x_s >= 20, name="min_scooters")
    # Non-negativity is implicit in variable definition

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the maximum number of meals delivered and vehicle counts
        return {
            "max_meals": m.objVal,
            "bikes": x_b.X,
            "scooters": x_s.X,
            "peak_system": int(y_peak.X),
        }
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_meal_delivery()
    if result is not None:
        print(f"Maximum Meals Delivered: {result['max_meals']}")
        print(f"Bikes used: {result['bikes']}")
        print(f"Scooters used: {result['scooters']}")
        print(f"Peak system activated (1=yes, 0=no): {result['peak_system']}")
    else:
        print("No feasible solution found.")