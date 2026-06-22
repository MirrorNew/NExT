def optimize_transportation():
    from gurobipy import Model, GRB, quicksum

    # Create a new model
    m = Model("TransportationOptimization")

    # Decision variables
    T = m.addVar(vtype=GRB.INTEGER, name="TruckTrips", lb=0, ub=5)
    C = m.addVar(vtype=GRB.INTEGER, name="CarTrips", lb=0)

    # ❤ Non-linearity is introduced. ❤
    # m.setObjective(20 * T + 15 * C, GRB.MINIMIZE)

    # ----- Introduce non-linear (piecewise) behavior for car fuel -----
    # Let:
    #   C1 = number of first 10 car trips (at 15 L per trip)
    #   C2 = number of car trips above 10 (from 11th on, at 20 L per trip)
    # Such that:
    #   C = C1 + C2
    #   C1 <= 10
    # This makes the car fuel cost piecewise linear:
    #   Car fuel = 15 * C1 + 20 * C2
    # which is equivalent to the original non-linear description.

    C1 = m.addVar(vtype=GRB.INTEGER, name="CarTripsFirst10", lb=0, ub=10)
    C2 = m.addVar(vtype=GRB.INTEGER, name="CarTripsAbove10", lb=0)

    # Link C with C1 and C2
    m.addConstr(C == C1 + C2, name="CarTripSplit")

    # Set objective: minimize total gas consumption with piecewise car cost
    m.setObjective(20 * T + 15 * C1 + 20 * C2, GRB.MINIMIZE)

    # Add constraints
    # Capacity constraint
    m.addConstr(50 * T + 30 * C >= 500, name="PackageRequirement")

    # Trip ratio constraint: at least 30% of all trips must be by car
    # C >= 0.3 * (T + C)  →  0.7C >= 0.3T  →  7C >= 3T
    m.addConstr(7 * C >= 3 * T, name="CarTripRatio")

    # T is at most 5 (already set as ub=5)
    # T >= 0 (lb=0), C >= 0 (lb=0)

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal total gas consumption and decision variables
        return {
            "MinTotalGas": m.objVal,
            "TruckTrips": T.X,
            "CarTrips": C.X,
            "CarTripsFirst10": C1.X,
            "CarTripsAbove10": C2.X,
        }
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_transportation()
    if result is not None:
        print(f"Minimum Total Gas Consumption: {result['MinTotalGas']}")
        print(f"Truck Trips: {result['TruckTrips']}")
        print(f"Car Trips: {result['CarTrips']}")
        print(f"  - First 10 Car Trips (15 L each): {result['CarTripsFirst10']}")
        print(f"  - Above 10 Car Trips (20 L each): {result['CarTripsAbove10']}")
    else:
        print("No feasible solution found.")