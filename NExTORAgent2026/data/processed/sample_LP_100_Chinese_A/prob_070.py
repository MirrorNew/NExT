def optimize_meal_preps(
    max_calories=2000,
    protein_smoothie=2,      # kept but no longer used in the nonlinear objective
    protein_bar=7,           # kept but no longer used in the nonlinear objective
    calories_smoothie=300,
    calories_bar=250
):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("MealPrepsMaxNonlinearProteinUtilization")

    # Decision variables: number of smoothies (x) and protein bars (y)
    # Since quantities are discrete, define as integer variables
    x = model.addVar(name="smoothies", vtype=GRB.INTEGER, lb=0)
    y = model.addVar(name="bars", vtype=GRB.INTEGER, lb=0)

    # New decision variable: total quantity z = x + y
    z = model.addVar(name="total_items", vtype=GRB.INTEGER, ub=10)

    # Add the relationship constraint: y = 2x (exact equality, per changed description)
    # ❤ Non-linearity is introduced. ❤
    # model.addConstr(y >= 2 * x, name="relation")
    model.addConstr(y == 2 * x, name="relation")

    # Add caloric constraint
    model.addConstr(
        calories_smoothie * x + calories_bar * y <= max_calories,
        name="calorie_limit"
    )

    # Constraint linking z with x and y: z = x + y
    model.addConstr(z == x*x + y, name="total_quantity")

    # ❤ Non-linearity is introduced. ❤
    # Set the objective: maximize total protein (original linear objective)
    # total_protein = protein_smoothie * x + protein_bar * y
    # model.setObjective(total_protein, GRB.MAXIMIZE)

    # New nonlinear objective:
    # Maximize the protein utilization index: 0.1 * x * y * z
    # Gurobi can handle this as a non-convex quadratic/degree-3 polynomial
    model.setParam("NonConvex", 2)  # allow non-convex nonlinear optimization
    Y = model.addVar()
    model.addConstr(Y == x * y)
    model.setObjective(0.1 * Y * z, GRB.MAXIMIZE)

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        # Return optimal values for x, y, z and the objective value
        return {
            "smoothies": x.X,
            "bars":      y.X,
            "total":     z.X,
            "max_protein_utilization": model.objVal
        }
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_meal_preps()
    if result is not None:
        print(f"Optimal number of smoothies (x): {result['smoothies']}")
        print(f"Optimal number of protein bars (y): {result['bars']}")
        print(f"Total items (z = x + y): {result['total']}")
        print(
            "Maximum nonlinear protein utilization index "
            f"(0.1 * x * y * z): {result['max_protein_utilization']:.2f}"
        )
    else:
        print("No feasible solution found.")