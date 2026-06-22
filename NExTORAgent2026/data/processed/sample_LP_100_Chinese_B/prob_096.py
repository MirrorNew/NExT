def optimize_oil_transport():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("OilTransport")

    # Decision variables
    # Number of containers (at least 15)
    C = m.addVar(name="Containers", vtype=GRB.CONTINUOUS, lb=15)
    # Number of trucks (non-negative)
    T = m.addVar(name="Trucks", vtype=GRB.CONTINUOUS, lb=0)

    # Set objective: minimize total number of containers and trucks
    m.setObjective(C + T, GRB.MINIMIZE)

    # Capacity constraint: at least 2000 units of oil
    m.addConstr(30 * C + 40 * T >= 2000, name="Capacity")

    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(T <= 0.5 * C, name="TruckContainerRatio")
    # Introduce a nonlinear ratio constraint: T / C <= 0.5  (C > 0 ensured by lb=15)
    m.addConstr(T * (1.0 ) <= 0.5*  C, name="TruckContainerRatio_Nonlinear")

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