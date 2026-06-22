def optimize_meal_plan():
    from gurobipy import Model, GRB, quicksum

    # Create a new model
    m = Model("MealOptimization")

    # Decision variables: number of fish and chicken meals
    x = m.addVar(name="fish_meals", lb=0, vtype=GRB.INTEGER)
    y = m.addVar(name="chicken_meals", lb=0, vtype=GRB.INTEGER)

    # ==============================
    # Modeling non-linear iron uptake
    # ==============================
    #
    # Iron from chicken remains linear: 8 * y
    # Iron from fish now grows exponentially with the number of fish meals.
    # Description: first fish meal gives 12 units; each additional meal's
    # contribution is 1.3 times the previous one's contribution.
    #
    # If we enumerate fish meals i = 1..x, then the iron from fish is
    #   12 * (1.3^0 + 1.3^1 + ... + 1.3^(x-1))
    # but this requires non-linear dependence on x.
    #
    # To introduce explicit non-linearity and keep x integer-bounded
    # for modeling, we create auxiliary variables for the per-meal
    # iron contribution sequence and link them with multiplicative
    # relationships. This uses Gurobi's general non-linear capability.

    # Upper bound on fish meals for modeling purposes (can be adjusted)
    max_fish = 20

    # Contribution from each possible fish meal i (i = 1..max_fish)
    # c[i] represents the iron contributed by the i-th fish meal
    c = {}
    for i in range(1, max_fish + 1):
        c[i] = m.addVar(name=f"fish_iron_contrib_{i}", lb=0.0, vtype=GRB.CONTINUOUS)

    # Binary activation variables: z[i] = 1 if the i-th fish meal is "used"
    z = {}
    for i in range(1, max_fish + 1):
        z[i] = m.addVar(name=f"fish_meal_active_{i}", vtype=GRB.BINARY)

    # Total iron from fish (non-linear aggregate)
    fish_iron = m.addVar(name="fish_iron_total", lb=0.0, vtype=GRB.CONTINUOUS)

    # ==============================
    # Objective: minimize total fat
    # ==============================
    m.setObjective(7 * x + 10 * y, GRB.MINIMIZE)

    # ==============================
    # Constraints
    # ==============================

    # Protein constraint (still linear)
    m.addConstr(10 * x + 15 * y >= 130, name="protein_req")

    # Link integer x with binary activation variables z[i]
    # Ensure x equals the number of active fish meals
    m.addConstr(x == quicksum(z[i] for i in range(1, max_fish + 1)), name="link_x_z")

    # Enforce order: if meal i+1 is active, then meal i must be active
    for i in range(1, max_fish):
        m.addConstr(z[i + 1] <= z[i], name=f"sequential_activation_{i}")

    # First fish meal iron contribution: 12 units if active
    base_iron = 12.0
    m.addConstr(c[1] == base_iron * z[1], name="fish_iron_first")

    # Subsequent fish meals: each is 1.3 times previous contribution if active.
    # This introduces multiplicative non-linearity: c[i] = 1.3 * c[i-1] * z[i]
    amplification = 1.3
    for i in range(2, max_fish + 1):
        # ❤ Non-linearity is introduced. ❤
        # Non-linear constraint: c[i] == amplification * c[i-1] * z[i]
        m.addConstr(c[i] == amplification * c[i-1] * z[i], name=f"scale_c_{i}")

    # Sum up total iron from fish
    m.addConstr(fish_iron == quicksum(c[i] for i in range(1, max_fish + 1)),
                name="fish_iron_sum")

    # ❤ Non-linearity is introduced. ❤
    # Replace the original linear iron constraint with the non-linear fish_iron term
    # m.addConstr(12 * x + 8 * y >= 120, name="iron_req")
    m.addConstr(fish_iron + 8 * y >= 120, name="iron_req_nonlinear")

    # Preference constraint (still linear)
    m.addConstr(y >= 2 * x, name="chicken_pref")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal total fat intake and solution details
        solution = {
            "objective_value": m.objVal,
            "fish_meals": x.X,
            "chicken_meals": y.X,
            "fish_iron": fish_iron.X
        }
        return solution
    else:
        # No feasible solution
        return None


if __name__ == '__main__':
    result = optimize_meal_plan()
    print(result)