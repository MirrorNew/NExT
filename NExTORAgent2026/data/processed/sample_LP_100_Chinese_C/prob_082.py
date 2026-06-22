def optimize_water_transport():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("Water_Transport_Maximize")

    # Decision variables
    # x: number of small kegs
    # y: number of large kegs
    x = m.addVar(vtype=GRB.INTEGER, name="small_kegs")
    y = m.addVar(vtype=GRB.INTEGER, name="large_kegs")

    # ❤ Non-linearity is introduced. ❤
    # Original linear objective (commented out):
    # m.setObjective(40 * x + 100 * y, GRB.MAXIMIZE)

    # Introduce a binary variable to model the non-linear step cost:
    # z = 1 if x > 20 (i.e., 21 or more small kegs), else 0
    z = m.addVar(vtype=GRB.BINARY, name="overtime_indicator")

    # Link z with x using big-M style constraints:
    # If z = 0 → x ≤ 20
    m.addConstr(x <= 20 + 30 * z, "upper_bound_small_with_z")
    # If z = 1 → x ≥ 21
    m.addConstr(x >= 21 * z, "lower_bound_small_with_z")

    # Set objective:
    # Maximize total liters of water transported,
    # while subtracting a penalty (cost) of 500 when overtime (z=1) occurs.
    # This makes the objective non-linear in terms of the original description
    # (piecewise behavior), but it is modeled linearly via binary variable z.
    m.setObjective(40 * x + 100 * y - 500 * z, GRB.MAXIMIZE)

    # Add constraints
    m.addConstr(x <= 30, "max_small_kegs")
    m.addConstr(y <= 10, "max_large_kegs")
    m.addConstr(x >= 2 * y, "small_at_least_twice_large")
    m.addConstr(x + y <= 25, "total_kegs_limit")
    m.addConstr(y >= 5, "min_large_kegs")
    m.addConstr(x >= 0, "non_negative_small")
    m.addConstr(y >= 0, "non_negative_large")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return detailed results: small kegs, large kegs, overtime flag, objective value
        return {
            "small_kegs": int(x.X),
            "large_kegs": int(y.X),
            "overtime_used": int(z.X),
            "effective_objective_value": m.objVal,
            "water_liters": 40 * int(x.X) + 100 * int(y.X)
        }
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_water_transport()
    if result is not None:
        print(f"Optimal number of small kegs: {result['small_kegs']}")
        print(f"Optimal number of large kegs: {result['large_kegs']}")
        print(f"Overtime fee incurred (1=yes, 0=no): {result['overtime_used']}")
        print(f"Total water transported (liters): {result['water_liters']}")
        print(f"Objective value (liters - 500*overtime): {result['effective_objective_value']}")
    else:
        print("No feasible solution found.")