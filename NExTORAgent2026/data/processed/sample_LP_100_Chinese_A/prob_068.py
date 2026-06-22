def optimize_meal_plan():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("MealOptimization_Nonlinear")

    # Decision variables: number of crab cakes and lobster rolls
    x = m.addVar(name="crab_cakes", lb=0, vtype=GRB.INTEGER)
    y = m.addVar(name="lobster_rolls", lb=0, vtype=GRB.INTEGER)

    # Set the objective: minimize total unsaturated fat
    m.setObjective(4 * x + 6 * y, GRB.MINIMIZE)

    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(5 * x + 8 * y >= 80, name="VitaminA")
    # Add vitamin A constraint with exponential absorption from crab cakes:
    # Effective vitamin A from crab cakes: 5 * (1.3 ** x)
    Y = m.addVar()
    m.addGenConstrPow(x,Y,1.3)
    m.addConstr(5 * Y + 8 * y >= 80, name="VitaminA_Nonlinear")

    # Add vitamin C constraint (unchanged, linear)
    m.addConstr(7 * x + 4 * y >= 100, name="VitaminC")

    # Add meal composition constraint (lobster at most 40%)
    m.addConstr(y <= (2 / 3) * x, name="LobsterRatio")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal objective value and decision variables
        return m.objVal, x.X, y.X
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_meal_plan()
    if result is not None:
        min_fat, crab_cakes_opt, lobster_rolls_opt = result
        print(f"Minimum Total Unsaturated Fat: {min_fat}")
        print(f"Optimal number of crab cakes: {crab_cakes_opt}")
        print(f"Optimal number of lobster rolls: {lobster_rolls_opt}")
    else:
        print("No feasible solution found.")