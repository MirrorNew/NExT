def optimize_oil_transport():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("OilTransport")

    # Decision variables
    # Number of containers (at least 15)
    C = m.addVar(name="Containers", vtype=GRB.INTEGER, lb=15)
    # Number of trucks (non-negative)
    T = m.addVar(name="Trucks", vtype=GRB.INTEGER, lb=0)

    # ❤ Non-linearity is introduced. ❤
    # Binary variable indicating whether total transported oil exceeds 2200 units
    z = m.addVar(name="SafetySchemeActive", vtype=GRB.BINARY)

    # Set objective: minimize total number of containers and trucks
    m.setObjective(C + T, GRB.MINIMIZE)

    # Capacity constraint: at least 2000 units of oil
    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(30 * C + 40 * T >= 2000, name="Capacity")
    m.addConstr(30 * C + 40 * T >= 2000, name="CapacityMin")

    # ❤ Non-linearity is introduced. ❤
    # Introduce piecewise / big-M style linkage for "exceeds 2200" logic
    # If total oil > 2200 (i.e., >= 2201), then z must be 1.
    # If total oil <= 2200, then z can be 0.
    # Use a sufficiently large M to bind the implication.
    M = 10000

    # Enforce: if z = 0, then 30C + 40T <= 2200
    m.addConstr(30 * C + 40 * T <= 2200 + M * z, name="UpperBoundOrActive")

    # Enforce: if z = 1, then 30C + 40T >= 2201
    m.addConstr(30 * C + 40 * T >= 2201 - M * (1 - z), name="LowerBoundIfActive")

    # ❤ Non-linearity is introduced. ❤
    # Additional container requirement when safety scheme is active:
    # If z = 1 (total oil ≥ 2201), then C ≥ 15 + 3 = 18
    # If z = 0, this reduces to the original C ≥ 15 requirement.
    m.addConstr(C >= 15 + 3 * z, name="ExtraContainersWhenActive")

    # Truck-to-container ratio constraint
    m.addConstr(T <= 0.5 * C, name="TruckContainerRatio")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal total number of containers and trucks
        return m.objVal
    else:
        # No feasible solution found
        return None


# Example usage
if __name__ == "__main__":
    min_transport = optimize_oil_transport()
    if min_transport is not None:
        print(f"Minimum Total Transport (Containers + Trucks): {min_transport}")
    else:
        print("No feasible solution found.")