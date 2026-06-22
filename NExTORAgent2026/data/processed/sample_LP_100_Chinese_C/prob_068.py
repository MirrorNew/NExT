def optimize_meal_plan():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("MealOptimization")

    # Decision variables: number of crab cakes and lobster rolls
    x = m.addVar(name="crab_cakes", lb=0, vtype=GRB.INTEGER)
    y = m.addVar(name="lobster_rolls", lb=0, vtype=GRB.INTEGER)

    # ❤ Non-linearity is introduced. ❤
    # Set the objective: minimize total unsaturated fat
    # m.setObjective(4 * x + 6 * y, GRB.MINIMIZE)

    # ❤ Non-linearity is introduced. ❤
    # New objective: if y > 10, an extra 20 units of fat from soup are added.
    # This introduces a piecewise / step behavior (non-linear in the original description).
    M = 1000  # a sufficiently large constant (Big-M)
    z = m.addVar(name="soup_indicator", vtype=GRB.BINARY)

    # Link z to the condition "y > 10":
    # If z = 0, enforce y <= 10.
    # If z = 1, y can be > 10 (up to an upper bound implied by other constraints / M).
    m.addConstr(y <= 10 + M * z, name="SoupTriggerUpper")
    m.addConstr(y >= 11 * z, name="SoupTriggerLower")

    # Objective now includes the possible 20 units of fat from the soup:
    # 4*x + 6*y + 20*z
    m.setObjective(4 * x + 6 * y + 20 * z, GRB.MINIMIZE)

    # Add vitamin A constraint
    m.addConstr(5 * x + 8 * y >= 80, name="VitaminA")
    # Add vitamin C constraint
    m.addConstr(7 * x + 4 * y >= 100, name="VitaminC")
    # Add meal composition constraint (lobster at most 40%)
    m.addConstr(y <= (2/3) * x, name="LobsterRatio")

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