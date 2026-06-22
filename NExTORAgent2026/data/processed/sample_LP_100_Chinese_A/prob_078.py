def minimize_animals(
    bricks_required=1000,
    bricks_per_cow=20,
    bricks_per_elephant=50
):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("Minimize Animals")

    # Decision variables: number of cows and elephants
    x = model.addVar(vtype=GRB.INTEGER, name="cows", lb=0)
    y = model.addVar(vtype=GRB.INTEGER, name="elephants", lb=0)

    # Set the objective: minimize total animals
    model.setObjective(x + y, GRB.MINIMIZE)

    # Add constraints
    # Brick transportation constraint
    model.addConstr(bricks_per_cow * x + bricks_per_elephant * y >= bricks_required, "bricks_constraint")
    # Elephants cannot exceed cows
    model.addConstr(y <= x, "elephants_not_exceed_cows")

    # ❤ Non-linearity is introduced. ❤
    # model.addConstr(x <= 2 * y, "cows_at_most_twice_elephants")
    # New nonlinear constraint:
    # (x / y)^1.2 <= 2  →  x^1.2 <= 2 * y^1.2  (for x, y > 0)
    # To model the ratio meaningfully, we ensure y >= 1
    y.lb = 1

    # Add a general (non-convex) power constraint: x^1.2 <= 2 * y^1.2
    # Using Gurobi's general constraints with exponent 1.2
    # We introduce two continuous auxiliary variables for x^1.2 and y^1.2
    x_pow = model.addVar(vtype=GRB.CONTINUOUS, name="x_pow", lb=0.0)
    y_pow = model.addVar(vtype=GRB.CONTINUOUS, name="y_pow", lb=0.0)

    # General power constraints: x_pow = x^1.2, y_pow = y^1.2
    model.addGenConstrPow(x, x_pow, 1.2, name="x_to_1_2")
    model.addGenConstrPow(y, y_pow, 1.2, name="y_to_1_2")

    # Nonlinear ratio constraint: x_pow <= 2 * y_pow
    model.addConstr(x_pow <= 2 * y_pow, name="nonlinear_cow_elephant_ratio")
    model.addConstr(x_pow >= 1.1 * y_pow, name="nonlinear_cow_elephant_ratio")

    # IMPORTANT: allow non-convex general constraints (since exponent 1.2 > 1)
    model.Params.NonConvex = 2
    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        total_animals = x.X + y.X
        print("Total animals: ", total_animals)
        print("Total cows: ", x.X)
        print("Total elephants: ", y.X)
        return total_animals
    else:
        return None


# Example usage
if __name__ == "__main__":
    total_animals = minimize_animals()
    if total_animals is not None:
        print(f"Minimum total number of animals: {total_animals}")
    else:
        print("No feasible solution found.")