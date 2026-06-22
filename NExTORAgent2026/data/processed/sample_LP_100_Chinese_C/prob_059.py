def optimize_pain_killers():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("PainKillerOptimization")

    # Decision variables: doses of pain killer 1 and 2
    x1 = m.addVar(name="PainKiller1", lb=0, vtype=GRB.INTEGER)
    x2 = m.addVar(name="PainKiller2", lb=0, vtype=GRB.INTEGER)

    # ❤ Non-linearity is introduced. ❤
    # Set the original (linear) objective: maximize back medicine
    # m.setObjective(0.8 * x1 + 0.4 * x2, GRB.MAXIMIZE)

    # ❤ Non-linearity is introduced. ❤
    # Introduce a binary variable to represent whether PainKiller2 dose exceeds 10
    y = m.addVar(name="Over10_PainKiller2", vtype=GRB.BINARY)

    # Objective: maximize effective back medicine after possible 2-unit deduction
    # Base back medicine: 0.8 * x1 + 0.4 * x2
    # Deduction: 2 units if x2 > 10 (captured by y == 1)
    m.setObjective(0.8 * x1 + 0.4 * x2 - 2 * y, GRB.MAXIMIZE)

    # Add constraints
    # Sleep medicine constraint
    m.addConstr(0.3 * x1 + 0.6 * x2 <= 8, name="SleepLimit")
    # Leg medicine constraint
    m.addConstr(0.5 * x1 + 0.7 * x2 >= 4, name="LegRequirement")

    # ❤ Non-linearity is introduced. ❤
    # Link y with x2 to represent the threshold condition "x2 > 10"
    # Use a big-M formulation:
    #   If y = 0  => enforce x2 <= 10
    #   If y = 1  => x2 can be larger than 10
    # Choose a sufficiently large M
    M = 100

    # When y = 0, this constraint forces x2 <= 10
    m.addConstr(x2 <= 10 + M * y, name="Link_y_upper")

    # When y = 1, this constraint allows x2 >= 11 (i.e., exceeding 10)
    # and when y = 0, it is nonbinding (x2 >= 0 is already implied by lb=0)
    m.addConstr(x2 >= 11 * y, name="Link_y_lower")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal doses and the final effective back medicine amount
        x1_opt = x1.X
        x2_opt = x2.X
        y_opt = y.X
        # Effective back medicine = objective value already includes deduction
        back_effective = m.objVal
        return {
            "PainKiller1_dose": x1_opt,
            "PainKiller2_dose": x2_opt,
            "Over10_flag": y_opt,
            "Effective_back_medicine": back_effective,
        }
    else:
        # No feasible solution found
        return None


if __name__ == '__main__':
    result = optimize_pain_killers()
    print(result)