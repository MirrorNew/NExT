from gurobipy import Model, GRB

def optimize_transportation():
    # Create a new model
    m = Model("TransportationOptimization")

    # Decision variables
    b = m.addVar(vtype=GRB.INTEGER, name="bikes", lb=0)
    c = m.addVar(vtype=GRB.INTEGER, name="cars", lb=0)

    # Set objective: minimize number of bikes
    m.setObjective(b, GRB.MINIMIZE)

    # Add capacity constraint
    m.addConstr(3 * b + 5 * c >= 500, name="capacity_constraint")

    # Add vehicle ratio constraint
    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(c <= (2/3) * b, name="vehicle_ratio_constraint")
    # New non-linear ratio constraint: c / (b + c) <= 2/5
    # i.e., cars / (bikes + cars) <= 0.4
    # This is non-linear due to the division by (b + c).
    m.addConstr(c <= 2 / 5 * (b + c), name="vehicle_ratio_constraint_nonlinear")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the minimal number of bikes and cars
        return int(b.X), int(c.X)
    else:
        # No feasible solution
        return None

# Example usage
if __name__ == "__main__":
    result = optimize_transportation()
    if result is not None:
        min_bikes, cars = result
        print(f"Minimum number of bikes: {min_bikes}")
        print(f"Number of cars: {cars}")
    else:
        print("No feasible solution found.")