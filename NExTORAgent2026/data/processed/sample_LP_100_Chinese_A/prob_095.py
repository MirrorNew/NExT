def optimize_transportation():
    from gurobipy import Model, GRB  # pow_ is used for non-linear (power) terms

    # Create a new model
    m = Model("TransportationOptimization")

    # Decision variables
    T = m.addVar(vtype=GRB.INTEGER, name="TruckTrips", lb=0, ub=5)
    C = m.addVar(vtype=GRB.INTEGER, name="CarTrips", lb=0)

    # ❤ Non-linearity is introduced. ❤
    # m.setObjective(20 * T + 15 * C, GRB.MINIMIZE)
    # Set objective: minimize total gas consumption
    # Truck: 20 liters per trip (linear), Car: 15^1.2 liters per trip (non-linear but constant per trip)
    Y = m.addVar()
    m.addConstr(Y == 15.0 *C)
    YY = m.addVar()
    m.addGenConstrPow(Y,YY,1.02)
    m.setObjective(20 * T + YY, GRB.MINIMIZE)

    # Add constraints
    # Capacity constraint
    m.addConstr(50 * T + 30 * C >= 500, name="PackageRequirement")
    # Trip ratio constraint
    m.addConstr(C >= (3 / 7) * T, name="CarTripRatio")
    # T is at most 5 (already set as ub=5)
    # T >= 0 (lb=0), C >= 0 (lb=0)

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal total gas consumption and decision variables
        return m.objVal, T.X, C.X
    else:
        # No feasible solution
        return None, None, None


# Example usage
if __name__ == "__main__":
    min_gas_consumption, opt_T, opt_C = optimize_transportation()
    if min_gas_consumption is not None:
        print(f"Minimum Total Gas Consumption: {min_gas_consumption}")
        print(f"Optimal Truck Trips (T): {opt_T}")
        print(f"Optimal Car Trips (C): {opt_C}")
    else:
        print("No feasible solution found.")