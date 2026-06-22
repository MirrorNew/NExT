def optimize_pain_killers():
    from gurobipy import Model, GRB, quicksum

    # Create a new model
    m = Model("PainKillerOptimization_Nonlinear")

    # Decision variables: doses of pain killer 1 and 2
    # Keep as integers as in the original model
    x1 = m.addVar(name="PainKiller1", lb=0, vtype=GRB.INTEGER)
    x2 = m.addVar(name="PainKiller2", lb=0, vtype=GRB.INTEGER)

    # Linear expression for total back medicine B
    B = 0.8 * x1 + 0.4 * x2

    # ---------------------------------------
    # Objective
    # ---------------------------------------

    # ❤ Non-linearity is introduced. ❤
    # m.setObjective(0.8 * x1 + 0.4 * x2, GRB.MAXIMIZE)

    # Introduce a general (non-linear) objective: maximize B^1.2
    # Note: Requires Gurobi to be used in a context that allows general functions.
    Y = m.addVar(name="PainKiller3", lb=0)
    K = m.addVar(name="PainKiller4", lb=0)
    m.addConstr(K == B, name="PainKiller5")
    m.addGenConstrPow(K ,Y,0.98)
    m.setObjective(Y, GRB.MAXIMIZE)

    # ---------------------------------------
    # Constraints
    # ---------------------------------------

    # Sleep medicine constraint
    m.addConstr(0.3 * x1 + 0.6 * x2 <= 8, name="SleepLimit")

    # Leg medicine constraint
    m.addConstr(0.5 * x1 + 0.7 * x2 >= 4, name="LegRequirement")

    # Optimize the model
    m.optimize()

    # Prepare result structure
    result = {
        "status": m.status,
        "objective_value": None,
        "x1": None,
        "x2": None,
        "back_total_B": None,
        "back_effective_B12": None,
    }

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        x1_opt = x1.X
        x2_opt = x2.X
        B_val = 0.8 * x1_opt + 0.4 * x2_opt
        B12_val = B_val ** 1.2

        result["objective_value"] = m.objVal
        result["x1"] = x1_opt
        result["x2"] = x2_opt
        result["back_total_B"] = B_val
        result["back_effective_B12"] = B12_val

    return result


if __name__ == "__main__":
    res = optimize_pain_killers()
    from gurobipy import GRB

    if res["status"] == GRB.OPTIMAL:
        print("Optimal solution found:")
        print(f"  PainKiller1 (x1) doses: {res['x1']}")
        print(f"  PainKiller2 (x2) doses: {res['x2']}")
        print(f"  Total back medicine B: {res['back_total_B']}")
        print(f"  Effective back medicine B^1.2: {res['back_effective_B12']}")
        print(f"  Reported objective value: {res['objective_value']}")
    else:
        print("No optimal solution found. Status code:", res["status"])