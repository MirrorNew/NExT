
def minimize_inheritance_difference():
    from gurobipy import Model, GRB
    import math

    # Item base values (P)
    values = {
        'painting': 25000,
        'bust': 5000,
        'vase': 20000,
        'porsche': 40000,
        'diamond1': 12000,
        'diamond2': 12000,
        'diamond3': 12000,
        'sofa': 3000,
        'dog1': 3000,
        'dog2': 3000,
        'sculpture': 10000,
        'boat': 15000,
        'motorcycle': 10000,
        'cavour_furniture': 13000
    }

    # Constants for time-dependent valuation
    n = 12
    pi_approx = 3.14159



    # ❤ Non-linearity is introduced. ❤
    # Original (time-independent) total value computation was:
    # V_total = sum(values.values())
    #
    # We now compute an effective value for each item in month t:
    # P_effective = P * [1 + 0.1 * cos(π * t / n)]
    # Note: cos introduces non-linearity with respect to parameter t in the valuation.


    # Initialize model
    m = Model("InheritancePartition_TimeDependent")

    Month_t = m.addVar(vtype=GRB.INTEGER, name='d',ub=12,lb=0)
    # Decision variables: y_i for each item
    Y = m.addVar(vtype=GRB.CONTINUOUS, name='d')
    Y2 = m.addVar(vtype=GRB.CONTINUOUS, name='d')
    m.addConstr(Y == pi_approx * Month_t / n)
    m.addGenConstrCos(Y,Y2)
    fluctuation_factor = 1 + 0.125 * Y2
    effective_values = {item: P * fluctuation_factor for item, P in values.items()}
    V_total = sum(effective_values.values())


    m.setParam('OutputFlag', 0)  # Silence output

    y_vars = {}
    for item in values:
        y_vars[item] = m.addVar(vtype=GRB.BINARY, name=f'y_{item}')

    # Auxiliary variable for absolute difference
    d = m.addVar(vtype=GRB.CONTINUOUS, name='d')

    m.update()

    # Objective: minimize d
    m.setObjective(d, GRB.MINIMIZE)

    # ❤ Non-linearity is introduced. ❤
    # Original constraint (time-independent values) was:
    # sum_values_y = sum(values[item] * y_vars[item] for item in values)
    # m.addConstr(2 * sum_values_y - V_total <= d, "abs_diff_upper")
    # m.addConstr(-(2 * sum_values_y - V_total) <= d, "abs_diff_lower")
    #
    # We now use month-t effective values instead of base values:
    sum_values_y = sum(effective_values[item] * y_vars[item] for item in effective_values)
    m.addConstr(2 * sum_values_y - V_total <= d, "abs_diff_upper")
    m.addConstr(-(2 * sum_values_y - V_total) <= d, "abs_diff_lower")

    # Constraint: dogs must stay together
    m.addConstr(y_vars['dog1'] == y_vars['dog2'], "dogs_together")

    # Optimize
    m.optimize()

    if m.status == GRB.OPTIMAL:
        # Return the minimal difference for month t
        return m.objVal,Month_t.X
    else:
        return None


if __name__ == "__main__":
    # Example: compute minimal inheritance difference for a given month t
    # You can change t to any integer from 1 to 12.
    result, month_t = minimize_inheritance_difference()
    if result is not None:
        print(f"Month t = {month_t}")
        print(f"Minimal inheritance difference (effective values): {result}")
    else:
        print("No feasible solution found.")