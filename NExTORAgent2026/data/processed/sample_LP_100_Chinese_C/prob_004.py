def minimize_inheritance_difference():
    from gurobipy import Model, GRB

    # Item values
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

    # Initialize model
    m = Model("InheritancePartitionWithFee")
    m.setParam('OutputFlag', 0)  # Silence output

    # Decision variables: y_i for each item
    y_vars = {}
    for item in values:
        y_vars[item] = m.addVar(vtype=GRB.BINARY, name=f'y_{item}')

    # Auxiliary variable for absolute (final) difference
    d = m.addVar(vtype=GRB.CONTINUOUS, name='d')

    # Additional variables for fee logic (non-linearity introduction)
    # Sum of values for son 1 (the one getting items with y_i = 1)
    S1 = m.addVar(vtype=GRB.CONTINUOUS, name='S1')
    # Total inheritance value
    V_total = sum(values.values())
    # Sum for son 2 is implied: S2 = V_total - S1

    # Binary flags indicating whether each son pays the fee
    z1 = m.addVar(vtype=GRB.BINARY, name='z1')  # fee flag for son 1
    z2 = m.addVar(vtype=GRB.BINARY, name='z2')  # fee flag for son 2

    # Non-linear piece: effective fee values
    fee1 = m.addVar(vtype=GRB.CONTINUOUS, name='fee1')  # = 5000 * z1
    fee2 = m.addVar(vtype=GRB.CONTINUOUS, name='fee2')  # = 5000 * z2

    m.update()

    # Link S1 to item selections
    m.addConstr(S1 == sum(values[item] * y_vars[item] for item in values), "sum_S1")

    # Compute S2
    # (We don't need an explicit variable; we express it where needed as V_total - S1)

    # Objective: minimize d (difference of FINAL values after possible fees)
    m.setObjective(d, GRB.MINIMIZE)

    # ❤ Non-linearity is introduced. ❤
    # Original absolute difference (before fees) constraints are commented out:
    # sum_values_y = sum(values[item] * y_vars[item] for item in values)
    # m.addConstr(2 * sum_values_y - V_total <= d, "abs_diff_upper")
    # m.addConstr(-(2 * sum_values_y - V_total) <= d, "abs_diff_lower")

    # --- New constraints to model the fee rule and absolute difference after fees ---

    # Constants
    FEE = 5000
    THRESHOLD = 90000
    BIG_M = 200000  # sufficiently large upper bound for sums and diff

    # 1) Model whether each son's pre-fee value exceeds 90000
    #    If S1 > 90000  -> z1 = 1 (pays fee)
    #    If S1 <= 90000 -> z1 = 0 (no fee)
    # Using big-M style indicator constraints (relaxed to linear form):

    # Lower bound when fee applies: if z1 = 1 then S1 >= THRESHOLD + small_eps
    # We use "strictly exceeds" via small epsilon (here we use 1 as a proxy).
    eps = 1.0
    m.addConstr(S1 >= THRESHOLD + eps - BIG_M * (1 - z1), "S1_lower_if_fee")
    # Upper bound when fee does not apply: if z1 = 0 then S1 <= THRESHOLD
    m.addConstr(S1 <= THRESHOLD + BIG_M * z1, "S1_upper_if_no_fee")

    # For son 2: S2 = V_total - S1
    # If S2 > 90000  -> z2 = 1
    # If S2 <= 90000 -> z2 = 0
    S2_expr = V_total - S1
    m.addConstr(S2_expr >= THRESHOLD + eps - BIG_M * (1 - z2), "S2_lower_if_fee")
    m.addConstr(S2_expr <= THRESHOLD + BIG_M * z2, "S2_upper_if_no_fee")

    # 2) Link fee variables to fee flags:
    #    fee1 = FEE * z1, fee2 = FEE * z2
    m.addConstr(fee1 == FEE * z1, "fee1_def")
    m.addConstr(fee2 == FEE * z2, "fee2_def")

    # 3) Define final values after fee:
    #    F1 = S1 - fee1, F2 = S2 - fee2
    F1 = m.addVar(vtype=GRB.CONTINUOUS, name='F1')
    F2 = m.addVar(vtype=GRB.CONTINUOUS, name='F2')
    m.addConstr(F1 == S1 - fee1, "final_val_son1")
    m.addConstr(F2 == S2_expr - fee2, "final_val_son2")

    # 4) Absolute difference of final values: d >= |F1 - F2|
    # Linearization:
    m.addConstr(F1 - F2 <= d, "abs_final_upper_pos")
    m.addConstr(F2 - F1 <= d, "abs_final_upper_neg")

    # Constraint: dogs must stay together
    m.addConstr(y_vars['dog1'] == y_vars['dog2'], "dogs_together")

    # Optimize
    m.optimize()

    if m.status == GRB.OPTIMAL:
        # Return the minimal final difference
        return m.objVal
    else:
        return None


if __name__ == "__main__":
    result = minimize_inheritance_difference()
    if result is not None:
        print(f"Minimal inheritance difference (after possible fees): {result}")
    else:
        print("No feasible solution found.")