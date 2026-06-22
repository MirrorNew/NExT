def optimize_bottle_production(
    water_available=250000,
    min_glass_bottles=20,
    glass_bottle_volume=500,
    plastic_bottle_volume=750,
    plastic_ratio=3
):
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("WaterBottleOptimization")
    m.setParam('OutputFlag', 0)  # Suppress Gurobi output

    # Decision variables
    x = m.addVar(name="glass_bottles", vtype=GRB.INTEGER, lb=min_glass_bottles)
    y = m.addVar(
        name="plastic_bottles",
        vtype=GRB.INTEGER,
        lb=plastic_ratio * min_glass_bottles
    )

    # Set objective: maximize total bottles
    m.setObjective(x + y, GRB.MAXIMIZE)

    # Add water constraint
    m.addConstr(
        glass_bottle_volume * x + plastic_bottle_volume * y <= water_available,
        "WaterLimit"
    )

    # ❤ Non-linearity is introduced. ❤
    # m.addConstr(y >= plastic_ratio * x, "PlasticRatio")
    # New nonlinear ratio: y >= 3 * 1.3^(x/20) * x
    Y1 = m.addVar()
    m.addConstr(Y1 == x / 20.0)

    # Note: exp_x_over_20 represents e^(x/20), but we need 1.3^(x/20).
    # 1.3^(x/20) = exp( ln(1.3) * x / 20 )
    # So instead we define z = ln(1.3) * Y1, then t = exp(z), and use t in constraint.
    import math
    z = m.addVar(name="z", lb=-GRB.INFINITY)
    t = m.addVar(name="t", lb=0.0)

    m.addConstr(z == math.log(1.3) * Y1, name="Def_z")
    m.addGenConstrExp(z, t, name="Exp_z")

    # Plastic bottle lower bound: y >= 3 * t * x
    m.addConstr(y >= plastic_ratio * t * x, name="NonlinearPlasticRatio")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        total_bottles = m.objVal
        glass_bottles = int(x.X)
        plastic_bottles = int(y.X)
        return glass_bottles, plastic_bottles, total_bottles
    else:
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_bottle_production()
    if result is not None:
        g, p, total = result
        print(f"Optimal glass bottles:   {g}")
        print(f"Optimal plastic bottles: {p}")
        print(f"Maximum Total Number of Bottles: {total}")
    else:
        print("No feasible solution found.")