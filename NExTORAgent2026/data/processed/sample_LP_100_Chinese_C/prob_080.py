def optimize_trucks():
    from gurobipy import Model, GRB, quicksum

    # Create a new model
    m = Model("SnowRemovalOptimization")

    # Decision variables
    # S: number of small trucks
    # L: number of large trucks
    S = m.addVar(vtype=GRB.INTEGER, name="SmallTrucks")
    L = m.addVar(vtype=GRB.INTEGER, name="LargeTrucks")

    # ❤ Non-linearity is introduced. ❤
    # Original linear objective (commented out):
    # m.setObjective(30 * S + 50 * L, GRB.MAXIMIZE)

    # Introduce a piecewise (non-linear in description) effect via additional variables
    # Define an integer variable for the number of small trucks above 15
    Over15 = m.addVar(vtype=GRB.INTEGER, name="Over15SmallTrucks", lb=0)

    # Over15 = max(S - 15, 0)
    # Implement via linear constraints:
    m.addConstr(Over15 >= S - 15, name="Over15_def1")
    m.addConstr(Over15 <= S - 15 + 1000 * (1 - 0), name="Over15_def2")  # loose upper bound
    m.addConstr(Over15 <= S, name="Over15_def3")

    # Set objective: maximize effective total snow transported
    # Base: 30*S + 50*L
    # Penalty if S > 15: -5 * (S - 15) = -5 * Over15
    m.setObjective(30 * S + 50 * L - 5 * Over15, GRB.MAXIMIZE)

    # Add constraints
    # Labor constraint
    m.addConstr(2 * S + 4 * L <= 30, name="LaborConstraint")
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
        # Print the optimal numbers of trucks and the effective snow transported
        S_opt = S.X
        L_opt = L.X
        obj_val = m.objVal
        print(f"Optimal Small Trucks: {S_opt}")
        print(f"Optimal Large Trucks: {L_opt}")
        print(f"Maximum Effective Total Snow Transported: {obj_val}")
        return obj_val
    else:
        if m.status == GRB.INFEASIBLE:
            print("Model is infeasible. Please check the constraints.")
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    max_snow = optimize_trucks()
    if max_snow is not None:
        print(f"Maximum Effective Total Snow Transported (Objective Value): {max_snow}")
    else:
        print("No feasible solution found.")