def optimize_beverage_production():
    from gurobipy import Model, GRB

    # Data parameters
    demand = [15, 25, 35, 25]  # demand for weeks 1-4 (in 1000 boxes)
    capacity = [30, 40, 45, 20]  # production capacity for weeks 1-4 (in 1000 boxes)
    cost = [5.0, 5.1, 5.4, 5.5]  # base production cost per 1000 boxes (in thousand yuan)
    storage_cost = 0.2  # storage cost per 1000 boxes per week (in thousand yuan)
    overtime_fixed_cost = 3.0  # fixed cost to start overtime line per week (in thousand yuan)
    overtime_extra_unit_cost = 0.5  # extra variable cost per 1000 boxes for overtime part (in thousand yuan)
    overtime_threshold = 10.0  # threshold above weekly demand to trigger overtime (in 1000 boxes)

    # Create model
    m = Model("BeverageProduction")

    # Decision variables
    # Regular production (up to demand + threshold) in each week
    P_reg = m.addVars(4, lb=0, name="P_reg")
    # Overtime production (above demand + threshold) in each week
    P_ot = m.addVars(4, lb=0, name="P_ot")
    # Inventory at end of each week
    I = m.addVars(4, lb=0, name="Inventory")
    # Binary variable: whether overtime is used in week t
    y_ot = m.addVars(4, vtype=GRB.BINARY, name="UseOvertime")

    # Initial inventory constraint
    # I_0 = 0 (not a variable, but initial condition)

    # Inventory balance constraints
    for t in range(4):
        if t == 0:
            m.addConstr(
                P_reg[t] + P_ot[t] + 0 == demand[t] + I[t],
                name=f"Balance_week_{t+1}",
            )
        else:
            m.addConstr(
                I[t - 1] + P_reg[t] + P_ot[t] == demand[t] + I[t],
                name=f"Balance_week_{t+1}",
            )

    # Production capacity constraints
    for t in range(4):
        # Total production cannot exceed capacity
        m.addConstr(P_reg[t] + P_ot[t] <= capacity[t], name=f"Cap_week_{t+1}")

    # ❤ Non-linearity is introduced. ❤
    # Original linear production variables and capacity constraints are replaced
    # by a piecewise structure with overtime part and a binary indicator.

    # Link regular and overtime production to demand and overtime usage
    bigM = max(capacity) + max(demand) + overtime_threshold  # sufficiently large

    for t in range(4):
        # 1) Regular production is limited to demand + threshold
        m.addConstr(
            P_reg[t] <= demand[t] + overtime_threshold,
            name=f"RegLimit_week_{t+1}",
        )

        # 2) Overtime production only if total production exceeds demand + threshold:
        #    P_ot[t] >= P_reg[t] + P_ot[t] - (demand[t] + threshold)  when y_ot = 1
        #    But we linearize using typical big-M links:

        # If y_ot == 0, then no overtime is allowed
        m.addConstr(
            P_ot[t] <= bigM * y_ot[t],
            name=f"OT_Upper_Use_week_{t+1}",
        )

        # If total production does not exceed demand + threshold,
        # overtime must be zero. Enforce:
        # P_reg[t] + P_ot[t] - (demand[t] + threshold) <= bigM * y_ot[t]
        m.addConstr(
            P_reg[t] + P_ot[t] - (demand[t] + overtime_threshold)
            <= bigM * y_ot[t],
            name=f"TriggerOT_week_{t+1}",
        )

        # Additionally, overtime part is exactly the amount above (demand + threshold),
        # but not negative: P_ot[t] >= P_reg[t] + P_ot[t] - (demand[t] + threshold)
        # This rearranges to: demand[t] + threshold >= P_reg[t]
        # which is already ensured by RegLimit_week, so we keep the simpler pattern
        # and let optimization choose P_ot subject to costs.

        # Optional tighter constraint: total production minus (demand+threshold)
        # is not larger than overtime production:
        m.addConstr(
            P_reg[t] + P_ot[t] - (demand[t] + overtime_threshold) <= P_ot[t],
            name=f"OT_LowerBound_week_{t+1}",
        )

    # Objective function
    total_cost = 0
    for t in range(4):
        # ❤ Non-linearity is introduced. ❤
        # Original objective term:
        # total_cost += cost[t] * P[t] + storage_cost * I[t]

        # Base production cost on all production
        total_cost += cost[t] * (P_reg[t] + P_ot[t])
        # Extra cost on overtime production
        total_cost += overtime_extra_unit_cost * P_ot[t]
        # Fixed overtime startup cost if overtime is used
        total_cost += overtime_fixed_cost * y_ot[t]
        # Inventory holding cost
        total_cost += storage_cost * I[t]

    m.setObjective(total_cost, GRB.MINIMIZE)

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
        print(f"Optimal total cost: {result}")
    else:
        print("No feasible solution found.")