def optimize_meal_plan():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("MealOptimization")

    # Decision variables: number of crab cakes and lobster rolls
    x = m.addVar(name="crab_cakes", lb=0, vtype=GRB.INTEGER)
    y = m.addVar(name="lobster_rolls", lb=0, vtype=GRB.INTEGER)

    # Set the objective: minimize total unsaturated fat
    m.setObjective(4 * x + 6 * y, GRB.MINIMIZE)

    # Add vitamin A constraint
    m.addConstr(5 * x + 8 * y >= 80, name="VitaminA")
    # Add vitamin C constraint
    m.addConstr(7 * x + 4 * y >= 100, name="VitaminC")

    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(y <= (2/3) * x, name="LobsterRatio")
    # Introduce non-linear constraint: y / x <= 2/3  (i.e., 3*y <= 2*x^2 when multiplied by x)
    # Note: This is a deliberately non-linear reformulation per the instruction.
    m.addQConstr(3 * y * x <= 2 * x * x, name="LobsterRatio_Nonlinear")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal objective value
        return m.objVal
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    min_fat = optimize_meal_plan()
    if min_fat is not None:
        print(f"Minimum Total Unsaturated Fat: {min_fat}")
    else:
        print("No feasible solution found.")