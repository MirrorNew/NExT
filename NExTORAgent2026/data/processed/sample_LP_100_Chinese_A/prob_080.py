def optimize_trucks():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("SnowRemovalOptimization")

    # Decision variables
    # S: number of small trucks
    # L: number of large trucks
    S = m.addVar(vtype=GRB.INTEGER, name="SmallTrucks")
    L = m.addVar(vtype=GRB.INTEGER, name="LargeTrucks")

    # Auxiliary loading equipment quantity (fixed as z = 1)
    z = m.addVar(vtype=GRB.INTEGER,ub=3, name="LargeTrucks")
    XXX = m.addVar()
    # Set objective: maximize total snow transported
    # ❤ Non-linearity is introduced. ❤
    # m.setObjective(30 * S + 50 * L, GRB.MAXIMIZE)
    m.addConstr(XXX == S * L)
    m.setObjective(30 * S + 50 * L + 1 * XXX * z, GRB.MAXIMIZE)

    # Add constraints
    # Labor constraint
    m.addConstr(2 * S + 4 * L <= 50, name="LaborConstraint")
    # Minimum small trucks
    m.addConstr(S >= 10, name="MinSmallTrucks")
    # Minimum large trucks
    m.addConstr(L >= 3, name="MinLargeTrucks")
    # Relation between small and large trucks
    m.addConstr(S == 2 * L, name="TruckRelation")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the maximum total snow transported
        print("SmallTrucks:",S)
        print("LargeTrucks:",L)
        print("z:",z)
        return m.objVal
    else:
        if m.status == GRB.INFEASIBLE:
            print("Model is infeasible. Please check the constraints.")
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    max_snow = optimize_trucks()
    if max_snow is not None:
        print(f"Maximum Total Snow Transported: {max_snow}")
    else:
        print("No feasible solution found.")