def optimize_meal_plan():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("MealOptimization")

    # Decision variables: number of fish and chicken meals
    x = m.addVar(name="fish_meals", lb=0, vtype=GRB.INTEGER)
    y = m.addVar(name="chicken_meals", lb=0, vtype=GRB.INTEGER)

    # ❤ Non-linearity is introduced. ❤
    # Original linear objective (commented out due to new non-linear condition):
    # m.setObjective(7 * x + 10 * y, GRB.MINIMIZE)

    # Auxiliary variable for extra-fat fish meals beyond 6
    # z represents max(0, x - 6)
    z = m.addVar(name="extra_fish_meals", lb=0, vtype=GRB.INTEGER)

    # Link z with x so that z = max(0, x - 6)
    m.addConstr(z >= x - 6, name="extra_fish_lower")
    m.addConstr(z >= 0,       name="extra_fish_nonneg")

    # Total fat:
    # - First 6 fish meals: 7 fat units each
    # - From 7th fish meal on: 12 fat units each
    #   => Total fish fat = 7*x + 5*z  (since each extra fish has extra 5 units)
    # - Chicken meals: 10 fat units each
    m.setObjective(7 * x + 5 * z + 10 * y, GRB.MINIMIZE)

    # Add constraints
    # Protein constraint
    m.addConstr(10 * x + 15 * y >= 130, name="protein_req")
    # Iron constraint
    m.addConstr(12 * x + 8 * y >= 120, name="iron_req")
    # Preference constraint
    m.addConstr(y >= 2 * x, name="chicken_pref")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        fish_meals = x.X
        chicken_meals = y.X
        total_fat = m.objVal
        return fish_meals, chicken_meals, total_fat
    else:
        # No feasible solution
        return None


if __name__ == '__main__':
    result = optimize_meal_plan()
    if result is not None:
        fish_meals, chicken_meals, total_fat = result
        print(f"Optimal solution:")
        print(f"  Fish meals    : {fish_meals}")
        print(f"  Chicken meals : {chicken_meals}")
        print(f"  Total fat     : {total_fat}")
    else:
        print("No feasible solution found.")