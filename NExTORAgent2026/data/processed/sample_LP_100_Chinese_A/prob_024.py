def optimize_production():
    from gurobipy import Model, GRB, quicksum

    # Data from the problem
    prep_costs = {'A': 1000, 'B': 920, 'C': 800, 'D': 700}
    unit_costs = {'A': 20, 'B': 24, 'C': 16, 'D': 28}
    capacities = {'A': 900, 'B': 1000, 'C': 1200, 'D': 1600}
    total_units = 2000

    # Parameters for the non-linear (stepwise, exponentially decreasing) cost on machine A
    block_size = 100                  # each cost "block" is 100 units
    num_blocks_A = capacities['A'] // block_size  # number of such 100-unit blocks on A

    # Precompute per-unit and per-block costs for each block on A
    # Block k (1-based): per-unit cost = 20 * 0.9^(k-1)
    # So per-block cost = block_size * 20 * 0.9^(k-1)
    block_unit_cost_A = {k: unit_costs['A'] * (0.9 ** (k - 1)) for k in range(1, num_blocks_A + 1)}
    block_cost_A = {k: block_size * block_unit_cost_A[k] for k in range(1, num_blocks_A + 1)}

    # Initialize model
    m = Model("Production_Optimization")
    m.setParam('OutputFlag', 0)  # Suppress output

    # Decision variables
    y = m.addVars(prep_costs.keys(), vtype=GRB.BINARY, name='y')           # machine usage
    x = m.addVars(prep_costs.keys(), lb=0, vtype=GRB.INTEGER, name='x')    # production quantity

    # ❤ Non-linearity is introduced. ❤
    # Original linear unit cost term (commented out):
    # m.setObjective(
    #     sum(prep_costs[i] * y[i] + unit_costs[i] * x[i] for i in prep_costs),
    #     GRB.MINIMIZE)

    # For machine A, replace linear unit cost with stepwise, exponentially decreasing cost.
    # Introduce block usage variables for A: z_A_k = number of units produced in block k (max 100 each).
    z_A = m.addVars(range(1, num_blocks_A + 1), lb=0, ub=block_size,
                    vtype=GRB.INTEGER, name='z_A')

    # Link total production on A with its blocks
    m.addConstr(x['A'] == quicksum(z_A[k] for k in z_A), name='Link_A_blocks')

    # Enforce sequential filling of blocks on A: block k+1 can only be used if block k is full.
    # This requires binaries for block activation.
    b_A = m.addVars(range(1, num_blocks_A + 1), vtype=GRB.BINARY, name='b_A')

    for k in range(1, num_blocks_A + 1):
        # If block k is "on", it can hold up to block_size units; otherwise 0.
        m.addConstr(z_A[k] <= block_size * b_A[k], name=f'CapBlockA_{k}')

    for k in range(1, num_blocks_A):
        # To use block k+1, block k must be fully activated.
        # A simple enforcement: b_A[k+1] <= b_A[k]
        m.addConstr(b_A[k + 1] <= b_A[k], name=f'SeqBlockA_{k}')

    # If machine A is off (y['A'] = 0), then no blocks can be used
    for k in range(1, num_blocks_A + 1):
        m.addConstr(b_A[k] <= y['A'], name=f'UseBlocksOnlyIfAOn_{k}')

    # Objective function with non-linear (in original sense) cost on A modeled via piecewise linear blocks
    # Preparation costs for all machines
    prep_term = quicksum(prep_costs[i] * y[i] for i in prep_costs)

    # Variable production cost for B, C, D remains linear
    lin_var_cost_other = quicksum(unit_costs[i] * x[i] for i in ['B', 'C', 'D'])

    # Variable production cost for A as sum of block costs
    var_cost_A = quicksum(block_unit_cost_A[k] * z_A[k] for k in z_A)

    m.setObjective(prep_term + lin_var_cost_other + var_cost_A, GRB.MINIMIZE)

    # Constraints
    # Demand satisfaction
    m.addConstr(quicksum(x[i] for i in prep_costs) == total_units, name='Demand')

    # Capacity constraints linked with activation
    for i in prep_costs:
        m.addConstr(x[i] <= capacities[i] * y[i], name=f'Cap_{i}')

    # Optimize
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        total_cost = m.objVal
        # Optionally, you could also return a detailed solution, but the original code returned only total_cost
        return total_cost
    else:
        return None


if __name__ == "__main__":
    result = optimize_production()
    if result is not None:
        print(f"Optimal total cost for production: {result}")
    else:
        print("No feasible solution found.")