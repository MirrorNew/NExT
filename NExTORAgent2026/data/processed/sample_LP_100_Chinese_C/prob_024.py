def optimize_production():
    from gurobipy import Model, GRB

    # Data from the problem
    prep_costs = {'A': 1000, 'B': 920, 'C': 800, 'D': 700}
    unit_costs = {'A': 20, 'B': 24, 'C': 16, 'D': 28}  # base unit costs
    capacities = {'A': 900, 'B': 1000, 'C': 1200, 'D': 1600}
    total_units = 2000

    # Initialize model
    m = Model("Production_Optimization")
    m.setParam('OutputFlag', 0)  # Suppress output

    # Decision variables
    # y[i] = 1 if machine i is used; 0 otherwise
    y = m.addVars(prep_costs.keys(), vtype=GRB.BINARY, name='y')

    # ❤ Non-linearity is introduced. ❤
    # x = m.addVars(prep_costs.keys(), lb=0, vtype=GRB.INTEGER, name='x')

    # x[i] is the total production on machine i
    x = m.addVars(prep_costs.keys(), lb=0, vtype=GRB.INTEGER, name='x')

    # For machine C, split production into two parts:
    # xC1: first 0–1000 units at 16 yuan/unit
    # xC2: units beyond 1000 (if any) at 22 yuan/unit, up to capacity
    xC1 = m.addVar(lb=0, ub=1000, vtype=GRB.INTEGER, name='xC1')
    xC2 = m.addVar(lb=0, ub=200, vtype=GRB.INTEGER, name='xC2')  # 1200 - 1000 = 200

    # Link split variables with total production on C
    m.addConstr(x['C'] == xC1 + xC2, name='Link_C_total')

    # Capacity-like constraint for first 1000 units on C:
    # If C is not used (y['C'] = 0), then xC1 = 0.
    # If C is used (y['C'] = 1), xC1 can be at most 1000.
    m.addConstr(xC1 <= 1000 * y['C'], name='Cap_C_part1')

    # Capacity-like constraint for units beyond 1000 on C:
    # If C is not used, xC2 = 0.
    # If C is used, xC2 can be at most 200.
    m.addConstr(xC2 <= 200 * y['C'], name='Cap_C_part2')

    # Demand satisfaction
    m.addConstr(sum(x[i] for i in prep_costs) == total_units, name='Demand')

    # Capacity constraints linked with activation for all machines
    for i in prep_costs:
        m.addConstr(x[i] <= capacities[i] * y[i], name=f'Cap_{i}')

    # Objective function
    # ❤ Non-linearity is introduced. ❤
    # m.setObjective(
    #     sum(prep_costs[i] * y[i] + unit_costs[i] * x[i] for i in prep_costs),
    #     GRB.MINIMIZE)

    # For machines A, B, and D, cost remains linear with unit_costs.
    # For machine C, we use piecewise linear form:
    #   prep_costs['C'] * y['C'] + 16 * xC1 + 22 * xC2
    m.setObjective(
        # Setup and variable costs for A, B, D
        sum(
            prep_costs[i] * y[i] + unit_costs[i] * x[i]
            for i in prep_costs
            if i != 'C'
        )
        # Setup cost for C
        + prep_costs['C'] * y['C']
        # Variable costs for C, split into two segments
        + 16 * xC1
        + 22 * xC2,
        GRB.MINIMIZE
    )

    # Optimize
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        total_cost = m.objVal
        # Optionally, you can also return detailed solution info if needed
        return total_cost
    else:
        return None


if __name__ == "__main__":
    result = optimize_production()
    if result is not None:
        print(f"Optimal total cost for production: {result}")
    else:
        print("No feasible solution found.")