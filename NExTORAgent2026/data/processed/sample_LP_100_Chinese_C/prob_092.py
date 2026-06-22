def optimize_taxi_fleet():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("TaxiFleetOptimization")

    # Decision variables
    x = m.addVar(vtype=GRB.INTEGER, name="motorcycles", lb=0)
    y = m.addVar(vtype=GRB.INTEGER, name="sedans", lb=0)

    # ❤ Non-linearity is introduced. ❤
    # Set objective: maximize total earnings
    # m.setObjective(100 * x + 225 * y, GRB.MAXIMIZE)

    # Binary variable to model the fixed management cost for sedans > 10
    z = m.addVar(vtype=GRB.BINARY, name="sedan_fixed_cost_indicator")

    # Big-M constant for linking y and z
    M = 1000  # sufficiently large upper bound on number of sedans

    # Objective with fixed cost: 500 is paid if y > 10 (z = 1)
    m.setObjective(100 * x + 225 * y - 500 * z, GRB.MAXIMIZE)

    # Constraints linking y and z (model "if y > 10 then z = 1", "if y <= 10 then z = 0")
    # When z = 0 → y ≤ 10; when z = 1 → y can be up to M
    m.addConstr(y <= 10 + M * z, name="sedan_upper_link")
    # Ensure that if y >= 11 then z must be 1
    m.addConstr(y >= 11 * z, name="sedan_lower_link")

    # Add constraints
    # Capacity constraint
    m.addConstr(30 * x + 70 * y >= 1200, name="capacity")
    # Motorcycle proportion constraint (motorcycles at most 25% of total vehicles)
    m.addConstr(4 * x <= y, name="motorcycle_ratio")
    # Pollution constraint
    m.addConstr(4 * x + 15 * y <= 200, name="pollution")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Print solution details
        print(f"Optimal number of motorcycles: {int(x.X)}")
        print(f"Optimal number of sedans: {int(y.X)}")
        print(f"Fixed management cost triggered (z): {int(z.X)}")
        print(f"Maximum Total Earnings (after fixed cost): ${m.objVal}")
        # Return the optimal objective value
        return m.objVal
    else:
        # No feasible solution found
        print("No feasible solution found.")
        return None


# Example usage
if __name__ == "__main__":
    max_earnings = optimize_taxi_fleet()
    if max_earnings is not None:
        print(f"Maximum Total Earnings: ${max_earnings}")