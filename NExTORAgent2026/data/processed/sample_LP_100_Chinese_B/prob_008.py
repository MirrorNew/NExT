def optimize_beverage_production():
    from gurobipy import Model, GRB, QuadExpr

    # Data parameters
    demand = [15, 25, 35, 25]  # demand for weeks 1-4
    capacity = [30, 40, 45, 20]  # production capacity for weeks 1-4
    cost = [5.0, 5.1, 5.4, 5.5]  # production cost per 1000 boxes
    storage_cost = 0.2  # storage cost per 1000 boxes per week

    total_demand = sum(demand)

    # Create model
    m = Model("BeverageProduction")

    # Decision variables
    P = m.addVars(4, lb=0, name="Production")  # production in each week
    I = m.addVars(4, lb=0, name="Inventory")   # inventory at end of each week

    # Initial inventory constraint
    # I_0 = 0 (not a variable, but initial condition)
    # Inventory balance constraints
    for t in range(4):
        if t == 0:
            m.addConstr(P[t] + 0 == demand[t] + I[t],
                        name=f"Balance_week_{t+1}")
        else:
            m.addConstr(I[t - 1] + P[t] == demand[t] + I[t],
                        name=f"Balance_week_{t+1}")

    # Production capacity constraints
    for t in range(4):
        m.addConstr(P[t] <= capacity[t], name=f"Cap_week_{t+1}")

    # ❤ Non-linearity is introduced. ❤
    # m.setObjective(total_cost, GRB.MINIMIZE)

    # Objective function (unchanged: still linear in P and I)
    total_cost = 0
    for t in range(4):
        total_cost += cost[t] * P[t] + storage_cost * I[t]

    # Introduce non-linearity: enforce
    #   (total production) / (total demand) <= 1.35
    # via a quadratic penalty term in the objective.
    #
    # Let S = sum_t P[t]. We want S / total_demand <= 1.35,
    # i.e. S <= 1.35 * total_demand.
    # Introduce a penalty variable z >= 0 to represent violation:
    #   z >= S - 1.35 * total_demand,  z >= 0
    # and add a quadratic penalty on z in the objective: M * z^2.
    # When the ratio is within 1.35, z = 0 and the constraint is not active.
    # When violated, the quadratic cost makes such solutions very unattractive.
    z = m.addVar(lb=0, name="OverRatioViolation")

    S = sum(P[t] for t in range(4))
    m.addConstr(z >= S - 1.35 * total_demand, name="RatioSoftConstraint")

    # Penalty weight: choose a large enough coefficient to strongly
    # discourage violating the ratio constraint.
    penalty_weight = 1000.0

    quad_obj = QuadExpr()
    quad_obj.add(total_cost)
    quad_obj.add(penalty_weight * z * z)

    m.setObjective(quad_obj, GRB.MINIMIZE)

    # Optimize
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        return m.objVal
    else:
        return None


if __name__ == "__main__":  # pragma: no cover
    result = optimize_beverage_production()
    if result is not None:
        print(f"Optimal total cost (with non-linear penalty): {result}")
    else:
        print("No feasible solution found.")