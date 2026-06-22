def optimize_pain_killers():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("PainKillerOptimization")

    # Decision variables: doses of pain killer 1 and 2
    x1 = m.addVar(name="PainKiller1", lb=0, vtype=GRB.INTEGER)
    x2 = m.addVar(name="PainKiller2", lb=0, vtype=GRB.INTEGER)

    # Set the objective: maximize back medicine
    m.setObjective(0.8 * x1 + 0.4 * x2, GRB.MAXIMIZE)

    # Add constraints
    # Sleep medicine constraint
    m.addConstr(0.3 * x1 + 0.6 * x2 <= 8, name="SleepLimit")
    # Leg medicine constraint
    m.addConstr(0.5 * x1 + 0.7 * x2 >= 4, name="LegRequirement")

    # ❤ Non-linearity is introduced. ❤
    # Add fractional (non-linear) constraint:
    # (leg_total / back_total) >= 0.6
    # where:
    #   leg_total  = 0.5 * x1 + 0.7 * x2
    #   back_total = 0.8 * x1 + 0.4 * x2
    leg_total = 0.5 * x1 + 0.7 * x2
    back_total = 0.8 * x1 + 0.4 * x2
    m.addQConstr(leg_total * 1.0 >= 0.6 * back_total, name="LegBackRatio")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal value of the objective function
        return m.objVal
    else:
        # No feasible solution found
        return None


if __name__ == '__main__':
    print(optimize_pain_killers())