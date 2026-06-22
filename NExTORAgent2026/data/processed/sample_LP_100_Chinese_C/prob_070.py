def optimize_meal_preps(max_calories=2000, protein_smoothie=2, protein_bar=7,
                        calories_smoothie=300, calories_bar=250,
                        extra_calories=200, bar_threshold=10):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("MealPrepsMaxProtein")

    # Decision variables: number of smoothies (x) and protein bars (y)
    # Since quantities are discrete, define as integer variables
    x = model.addVar(name="smoothies", vtype=GRB.INTEGER, lb=0)
    y = model.addVar(name="bars", vtype=GRB.INTEGER, lb=0)

    # Add the relationship constraint: y >= 2x
    model.addConstr(y >= 2 * x, name="relation")

    # ❤ Non-linearity is introduced. ❤
    # Original simple calorie constraint is commented out and replaced below.
    # model.addConstr(
    #     calories_smoothie * x + calories_bar * y <= max_calories,
    #     name="calorie_limit"
    # )

    # Introduce an auxiliary binary variable to model the piecewise calorie effect
    # z = 1 if y > bar_threshold, 0 otherwise
    z = model.addVar(name="extra_side_dish", vtype=GRB.BINARY)

    # Big‑M style linking constraints to ensure:
    #   z = 0  ->  y <= bar_threshold
    #   z = 1  ->  y  can exceed bar_threshold
    M = 1000  # sufficiently large upper bound for number of bars

    # When z = 0, y <= bar_threshold
    model.addConstr(y <= bar_threshold + M * z, name="link_upper")

    # When z = 1, y >= bar_threshold + 1 (i.e., more than bar_threshold bars)
    model.addConstr(y >= bar_threshold + 1 - M * (1 - z), name="link_lower")

    # Calorie constraint including the extra fixed 200 calories when z = 1
    model.addConstr(
        calories_smoothie * x + calories_bar * y + extra_calories * z <= max_calories,
        name="calorie_limit_with_extra"
    )

    # Set the objective: maximize total protein
    total_protein = protein_smoothie * x + protein_bar * y
    model.setObjective(total_protein, GRB.MAXIMIZE)

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        smoothies_val = x.X
        bars_val = y.X
        total_protein_val = model.objVal
        return smoothies_val, bars_val, total_protein_val
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_meal_preps()
    if result is not None:
        smoothies, bars, max_protein = result
        print(f"Optimal number of smoothies: {smoothies:.0f}")
        print(f"Optimal number of protein bars: {bars:.0f}")
        print(f"Maximum protein intake: {max_protein:.2f} units")
    else:
        print("No feasible solution found.")