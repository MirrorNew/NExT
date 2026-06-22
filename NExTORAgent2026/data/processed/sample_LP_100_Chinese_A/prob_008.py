def optimize_beverage_production():
    from gurobipy import Model, GRB, QuadExpr

    # Data parameters
    demand = [15, 25, 35, 25]  # demand for weeks 1-4
    capacity = [30, 40, 45, 20]  # production capacity for weeks 1-4
    cost = [5.0, 5.1, 5.4, 5.5]  # production cost per 1000 boxes
    storage_cost = 0.2  # coefficient in the nonlinear storage cost term

    # Create model
    m = Model("BeverageProduction_Nonlinear")

    # Decision variables
    P = m.addVars(4, lb=0, name="Production")  # production in each week: P1..P4
    I = m.addVars(4, lb=0, name="Inventory")   # inventory at end of each week: I1..I4

    # Inventory balance constraints
    # I_0 = 0 (given initial condition, not a variable)
    for t in range(4):
        if t == 0:
            m.addConstr(P[t] == demand[t] + I[t],
                        name=f"Balance_week_{t+1}")
        else:
            m.addConstr(I[t - 1] + P[t] == demand[t] + I[t],
                        name=f"Balance_week_{t+1}")

    # Production capacity constraints
    for t in range(4):
        m.addConstr(P[t] <= capacity[t], name=f"Cap_week_{t+1}")

    # Objective function
    # Production cost (still linear)
    obj = QuadExpr()
    for t in range(4):
        obj.add(cost[t] * P[t])

    # ❤ Non-linearity is introduced. ❤
    # Previously: linear inventory cost  storage_cost * I[t]
    # for t in range(4):
    #     total_cost += cost[t] * P[t] + storage_cost * I[t]

    # Now: nonlinear inventory cost:
    # Week 1: 0.2 * I1 * P1 * I2
    # Week 2: 0.2 * I2 * P2 * I3
    # Week 3: 0.2 * I3 * P3 * I4
    # Week 4: 0.2 * I4 * P4 * I1
    # This is a polynomial with products of three variables.
    # Gurobi does not support general cubic terms directly in the objective,
    # so we introduce auxiliary variables and constraints to represent them.

    # Auxiliary variables for pairwise products I_t * P_t
    IP = m.addVars(4, lb=0, name="IP")  # IP_t = I_t * P_t

    # Define IP_t = I_t * P_t using general constraints
    for t in range(4):
        m.addConstr(IP[t] == I[t] * P[t], name=f"Def_IP_{t+1}")

    # Auxiliary variables for the full cubic terms: I_t * P_t * I_{t+1}
    C = m.addVars(4, lb=0, name="CubicCostTerm")

    # Link C_t = IP_t * I_{t+1}
    for t in range(4):
        next_t = (t + 1) % 4  # next week index (1→2, 2→3, 3→4, 4→1)
        m.addConstr(C[t] == IP[t] * I[next_t], name=f"Def_Cubic_{t+1}")

    # Add nonlinear inventory cost to the objective
    for t in range(4):
        obj.add(storage_cost * C[t])

    m.setObjective(obj, GRB.MINIMIZE)

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
        print(f"Optimal total cost (with nonlinear inventory cost): {result}")
    else:
        print("No feasible solution found.")