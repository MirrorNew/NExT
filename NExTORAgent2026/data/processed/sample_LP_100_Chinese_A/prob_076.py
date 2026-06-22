def optimize_vitamins():
    from gurobipy import Model, GRB, quicksum

    # Create a new model
    m = Model("VitaminIntakeMaximization_Nonlinear")

    # Decision variables: number of orange and apple juice boxes
    x = m.addVar(name="orange_boxes", vtype=GRB.INTEGER, lb=3)
    y = m.addVar(name="apple_boxes", vtype=GRB.INTEGER, lb=0)

    # ❤ Non-linearity is introduced. ❤
    # m.setObjective(10 * x + 12 * y, GRB.MAXIMIZE)

    # Nonlinear objective:
    # Total Vitamin D = 10 * x (from orange juice)
    #                 + (12 * y) ** 1.2 (nonlinear enhanced effect from apple juice)

    Y = m.addVar()
    m.addConstr(Y == 12 * y)
    YY = m.addVar()
    m.addGenConstrPow(Y,YY,1.02)

    m.setObjective(10 * x + YY, GRB.MAXIMIZE)

    # Add constraints
    # Vitamin C constraint
    m.addConstr(8 * x + 6 * y <= 300, name="VitaminC_limit")
    # Preference constraint: at least 3 times as many apple as orange
    m.addConstr(y >= 3 * x, name="Preference")
    # Minimum orange boxes
    m.addConstr(x >= 3, name="Min_orange")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the maximum vitamin D intake and the optimal numbers of boxes
        return {
            "max_vitamin_d": m.objVal,
            "orange_boxes": x.X,
            "apple_boxes": y.X
        }
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_vitamins()
    if result is not None:
        print(f"Maximum Vitamin D Intake: {result['max_vitamin_d']}")
        print(f"Optimal orange juice boxes: {result['orange_boxes']}")
        print(f"Optimal apple juice boxes: {result['apple_boxes']}")
    else:
        print("No feasible solution found.")