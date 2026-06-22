import math


def optimize_minty_foam():
    from gurobipy import Model, GRB, quicksum

    # Create a new model
    model = Model("MintyFoamOptimization_Nonlinear")

    # Decision variables: number of demonstrations (now integer and nonnegative)
    x1 = model.addVar(vtype=GRB.INTEGER, lb=0, name="Demo1")
    x2 = model.addVar(vtype=GRB.INTEGER, lb=1, name="Demo2")

    # ≈≈≈ Introduce auxiliary variables for nonlinearity handling ≈≈≈
    # Total number of experiments n = x1 + x2
    n = model.addVar(vtype=GRB.INTEGER, lb=0, name="TotalDemos")

    # Precompute maximum possible number of experiments from linear resource limits
    # 10*x1 + 12*x2 <= 120  and  20*x1 + 15*x2 <= 100
    # Upper bound for safety; small instance so we can afford an outer-approximation
    max_n = 12  # conservative bound from quick manual reasoning

    # Precompute 1.3^k for all possible k in [0, max_n]
    pow_vals = {k: 1.5 ** k for k in range(max_n + 1)}

    # Variable representing 1.3^n (will be linked to n via constraints)
    t = model.addVar(lb=0.0, name="Pow_1_3_n")

    # Variable for the linear black tar total: 5*x1 + 3*x2
    tar_linear = model.addVar(lb=0.0, name="Tar_Linear")

    # Variable for the effective tar: (5*x1 + 3*x2) * 1.3^n
    tar_effective = model.addVar(lb=0.0, name="Tar_Effective")

    # Set the objective: maximize total minty foam
    model.setObjective(25 * x1 + 18 * x2, GRB.MAXIMIZE)

    # Add resource constraints
    model.addConstr(10 * x1 + 12 * x2 <= 120, "MintConstraint")
    model.addConstr(20 * x1 + 15 * x2 <= 100, "ActiveIngredientConstraint")

    # Link total number of experiments
    model.addConstr(n == x1 + x2, "TotalExperiments")

    # ❤ Non-linearity is introduced. ❤
    # model.addConstr(5 * x1 + 3 * x2 <= 50, "BlackTarConstraint")

    # Linear black tar total
    model.addConstr(tar_linear == 5 * x1 + 3 * x2, "TarLinearDef")

    x1x2=model.addVar()
    model.addConstr(x1 + x2 == x1x2, "X1")
    ppp=model.addVar(lb=0, name="ppp")

    # ppp == 1.3^(x1 + x2)
    lnppp = model.addVar(lb=0, name="lnppp")
    model.addConstr(lnppp ==  x1x2 * math.log(1.03), "X1")
    model.addGenConstrLog(ppp,lnppp)

    # Nonlinear (approximated) black tar constraint:
    # (5*x1 + 3*x2) * 1.3^(x1 + x2) <= 50
    model.addConstr(tar_linear *  ppp<= 50, "BlackTarEffectiveConstraint")

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        x1_val = int(round(x1.X))
        x2_val = int(round(x2.X))
        obj_val = model.objVal
        return x1_val, x2_val, obj_val
    else:
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_minty_foam()
    if result is not None:
        x1_opt, x2_opt, max_minty_foam = result
        print(f"Optimal Demo1 count: {x1_opt}")
        print(f"Optimal Demo2 count: {x2_opt}")
        print(f"Maximum Total Minty Foam Produced: {max_minty_foam}")
    else:
        print("No feasible solution found.")