def minimize_animals(
    bricks_required=1000,
    bricks_per_cow=20,
    bricks_per_elephant=50
):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("Minimize Animals with Nonlinear-like Requirement")

    # Decision variables: number of cows and elephants
    x = model.addVar(vtype=GRB.INTEGER, name="cows", lb=0)
    y = model.addVar(vtype=GRB.INTEGER, name="elephants", lb=0)

    # ❤ Non-linearity is introduced. ❤
    # Binary variable to indicate whether planned bricks exceed 1200
    z = model.addVar(vtype=GRB.BINARY, name="extra_cows_trigger")

    # Planned transported bricks
    planned_bricks = bricks_per_cow * x + bricks_per_elephant * y

    # Set the objective: minimize total animals
    # ❤ Non-linearity is introduced. ❤
    # Original (commented out):
    # model.setObjective(x + y, GRB.MINIMIZE)
    # New objective: add 3 extra "virtual cows" when planned bricks > 1200 (captured by z)
    model.setObjective(x + y + 3 * z, GRB.MINIMIZE)

    # Add constraints
    # Brick transportation constraint (at least required bricks)
    model.addConstr(planned_bricks >= bricks_required, "bricks_constraint")

    # Elephants cannot exceed cows
    model.addConstr(y <= x, "elephants_not_exceed_cows")

    # At most twice the number of cows as elephants
    model.addConstr(x <= 2 * y, "cows_at_most_twice_elephants")

    # ❤ Non-linearity is introduced. ❤
    # Logical condition: if planned_bricks > 1200 then z = 1,
    # else we can keep z = 0 in the optimal solution.
    #
    # We use a big-M linearization:
    #   planned_bricks - 1200 <= M * z
    #   planned_bricks - 1200 >= 1 - M * (1 - z)
    #
    # This enforces:
    #   - when z = 0: planned_bricks <= 1200  (from first inequality)
    #   - when z = 1: planned_bricks >= 1201 (from second inequality)
    #
    # Choose a sufficiently large M based on reasonable upper bound of bricks.
    # A safe upper bound: if all animals were cows, say up to 200 animals,
    # max bricks ≈ 200 * 50 = 10000 (take M a bit larger).
    M = 10000

    model.addConstr(planned_bricks - 1200 <= M * z, name="trigger_upper")
    model.addConstr(planned_bricks - 1200 >= 1 - M * (1 - z), name="trigger_lower")

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        total_animals = x.X + y.X + 3 * z.X
        return total_animals
    else:
        return None


# Example usage
if __name__ == "__main__":
    total_animals = minimize_animals()
    if total_animals is not None:
        print(f"Minimum total number of animals (including extra 3 cows if triggered): {total_animals}")
    else:
        print("No feasible solution found.")