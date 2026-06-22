def optimize_vehicles():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("airport_vehicles")

    # Decision variables: number of 4-wheelers and 3-wheelers
    x = m.addVar(vtype=GRB.INTEGER, name="x")  # 4-wheeler
    y = m.addVar(vtype=GRB.INTEGER, name="y")  # 3-wheeler

    # Set the objective: minimize total number of vehicles
    m.setObjective(x + y, GRB.MINIMIZE)

    # Add constraints
    # Luggage capacity constraint
    m.addConstr(60 * x + 40 * y >= 1000, "luggage_constraint")
    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(30 * x + 15 * y <= 430, "pollution_constraint")

    # Non-linear pollution constraint:
    # (total pollution) / (total luggage capacity) <= 0.5
    # i.e., (30*x + 15*y) / (60*x + 40*y) <= 0.5
    # This is a fractional (non-linear) constraint being added as-is.
    m.addConstr((30 * x + 15 * y)  <= 0.5 *  (60 * x + 40 * y), "pollution_ratio_constraint")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Print optimal values of x and y for clarity
        x_val = x.X
        y_val = y.X
        print(f"Optimal number of 4-wheelers (x): {x_val}")
        print(f"Optimal number of 3-wheelers (y): {y_val}")
        return m.objVal
    else:
        return None


# Example usage
if __name__ == "__main__":
    min_vehicles = optimize_vehicles()
    if min_vehicles is not None:
        print(f"Minimum Total Number of Vehicles: {min_vehicles}")
    else:
        print("No feasible solution found.")