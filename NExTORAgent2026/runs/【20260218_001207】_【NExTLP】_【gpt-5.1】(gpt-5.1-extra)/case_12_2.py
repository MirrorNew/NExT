import gurobipy as gp
from gurobipy import GRB

# This script builds and solves the MILP using ONLY the given Parameters List.
# It also prints the final answer in the required format:
#   print(f"FinalAnswer=【{the_question_answer}】")

# -----------------------------
# 1. Define all parameter matrices and data inputs
# -----------------------------

# Basic scalar parameters
num_periods = 12
num_assets = 6
overall_risk_index_upper_bound = 6.0
max_stock_weight_per_period = 0.7
min_bond_weight_per_period = 0.3
max_weight_per_stock_asset = 0.4
max_weight_per_bond_asset = 0.6
min_weight_if_selected = 0.1
min_num_assets_per_period = 4
max_avg_total_weight_stock_Z = 0.4
max_avg_total_weight_bond_M = 0.4

# Table 1 data and lists
assets_list = ['Stock X', 'Stock Y', 'Stock Z', 'Bond M', 'Bond N', 'Bond O']
expected_annual_return_percent_original = [12, 10, 15, 5, 4, 6]
risk_index_per_asset = [8, 6, 10, 2, 1, 3]

# Period and asset short-name lists
periods_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
assets_short_names = ['X', 'Y', 'Z', 'M', 'N', 'O']

# Table 2: current expected annual return per (period, asset) in percent
current_expected_annual_return_percent = [
    [11.5, 9.8, 14.2, 5.1, 3.9, 6.2],
    [12.3, 10.2, 15.1, 4.9, 4.1, 5.8],
    [11.8, 9.5, 14.7, 5.0, 4.0, 6.0],
    [12.0, 10.0, 15.3, 5.2, 3.8, 6.1],
    [12.1, 10.1, None, 5.0, 4.2, 5.9],
    [11.9, 9.7, 15.0, 5.3, 4.0, 6.3],
    [12.4, 10.3, 15.2, 5.1, 4.1, 6.0],
    [12.2, 10.0, 14.9, 5.0, 3.9, 6.2],
    [11.7, 9.6, 14.5, 4.8, 4.0, 6.1],
    [12.5, 10.4, None, 5.2, 4.3, 5.7],
    [12.0, 10.1, 15.4, 5.1, 4.2, 6.0],
    [11.6, 9.9, 14.8, 5.0, 4.1, 6.2]
]

# Mapping short names to long asset names
short_to_long = dict(zip(assets_short_names, assets_list))

# Asset type partitions
stock_assets = ['Stock X', 'Stock Y', 'Stock Z']
bond_assets = ['Bond M', 'Bond N', 'Bond O']

# Risk index by asset (dictionary form)
risk_index = {assets_list[j]: risk_index_per_asset[j] for j in range(num_assets)}

# Uninvestable asset-period pairs: entries where Table 2 has None
uninvestable = set()
for t_idx, t in enumerate(periods_list):
    for a_short_idx, a_short in enumerate(assets_short_names):
        if current_expected_annual_return_percent[t_idx][a_short_idx] is None:
            a_long = short_to_long[a_short]
            uninvestable.add((a_long, t))

# Current expected return r_{i,t} for all (asset, period)
# For uninvestable entries, define r = 0.0 (they will be fixed to 0 weight anyway)
r = {}
for t_idx, t in enumerate(periods_list):
    for a_short_idx, a_short in enumerate(assets_short_names):
        a_long = short_to_long[a_short]
        val = current_expected_annual_return_percent[t_idx][a_short_idx]
        r[(a_long, t)] = 0.0 if val is None else float(val)

# -----------------------------
# 2. Create Gurobi model
# -----------------------------
model = gp.Model("Investment_Portfolio_Optimization")

# -----------------------------
# 3. Create decision variables
# -----------------------------

# w_{i,t}: continuous weight of asset i in period t
w = model.addVars(
    assets_list,
    periods_list,
    vtype=GRB.CONTINUOUS,
    lb=0.0,
    ub=1.0,
    name="w"
)

# y_{i,t}: binary selection variable (1 if asset i is invested in period t)
y = model.addVars(
    assets_list,
    periods_list,
    vtype=GRB.BINARY,
    name="y"
)

# n_t: integer number of assets invested in period t
n = model.addVars(
    periods_list,
    vtype=GRB.INTEGER,
    lb=0,
    ub=num_assets,
    name="n"
)

# -----------------------------
# 4. Set up the objective function
#     max Z = Σ_t Σ_i r_{i,t} * w_{i,t}
#     and we will derive the expected multiple from Z after optimization
# -----------------------------
obj_expr = gp.quicksum(r[(i, t)] * w[i, t] for i in assets_list for t in periods_list)
model.setObjective(obj_expr, GRB.MAXIMIZE)

# -----------------------------
# 5. Add all constraints
# -----------------------------

# (1) Period sum-to-one: Σ_i w_{i,t} = 1
for t in periods_list:
    model.addConstr(
        gp.quicksum(w[i, t] for i in assets_list) == 1.0,
        name=f"sum_to_one_t{t}"
    )

# (2) & (3) Uninvestable assets: w_{i,t} = 0, y_{i,t} = 0
for (i, t) in uninvestable:
    model.addConstr(w[i, t] == 0.0, name=f"uninv_w_{i}_{t}")
    model.addConstr(y[i, t] == 0, name=f"uninv_y_{i}_{t}")

# (4) Per-period risk limit: Σ_i k_i * w_{i,t} ≤ overall_risk_index_upper_bound
for t in periods_list:
    model.addConstr(
        gp.quicksum(risk_index[i] * w[i, t] for i in assets_list)
        <= overall_risk_index_upper_bound,
        name=f"risk_limit_t{t}"
    )

# (5) Single stock upper bound: w_{i,t} ≤ max_weight_per_stock_asset
for t in periods_list:
    for i in stock_assets:
        model.addConstr(
            w[i, t] <= max_weight_per_stock_asset,
            name=f"single_stock_ub_{i}_{t}"
        )

# (6) Single bond upper bound: w_{i,t} ≤ max_weight_per_bond_asset
for t in periods_list:
    for i in bond_assets:
        model.addConstr(
            w[i, t] <= max_weight_per_bond_asset,
            name=f"single_bond_ub_{i}_{t}"
        )

# (7) Per-period total stock fraction upper bound: Σ_{stocks} w_{i,t} ≤ max_stock_weight_per_period
for t in periods_list:
    model.addConstr(
        gp.quicksum(w[i, t] for i in stock_assets) <= max_stock_weight_per_period,
        name=f"stock_fraction_ub_t{t}"
    )

# (8) Per-period total bond fraction lower bound: Σ_{bonds} w_{i,t} ≥ min_bond_weight_per_period
for t in periods_list:
    model.addConstr(
        gp.quicksum(w[i, t] for i in bond_assets) >= min_bond_weight_per_period,
        name=f"bond_fraction_lb_t{t}"
    )

# (9) Minimum number of assets per period: Σ_i y_{i,t} ≥ min_num_assets_per_period
for t in periods_list:
    model.addConstr(
        gp.quicksum(y[i, t] for i in assets_list) >= min_num_assets_per_period,
        name=f"min_assets_t{t}"
    )

# (10) Definition of n_t: n_t = Σ_i y_{i,t}
for t in periods_list:
    model.addConstr(
        n[t] == gp.quicksum(y[i, t] for i in assets_list),
        name=f"define_n_t{t}"
    )

# (11) Average weight cap for Stock Z: (1/12) Σ_t w_{Z,t} ≤ max_avg_total_weight_stock_Z
#      -> Σ_t w_{Z,t} ≤ num_periods * max_avg_total_weight_stock_Z
model.addConstr(
    gp.quicksum(w['Stock Z', t] for t in periods_list)
    <= num_periods * max_avg_total_weight_stock_Z,
    name="avg_weight_cap_Z"
)

# (12) Average weight cap for Bond M: (1/12) Σ_t w_{M,t} ≤ max_avg_total_weight_bond_M
#      -> Σ_t w_{M,t} ≤ num_periods * max_avg_total_weight_bond_M
model.addConstr(
    gp.quicksum(w['Bond M', t] for t in periods_list)
    <= num_periods * max_avg_total_weight_bond_M,
    name="avg_weight_cap_M"
)

# (13) Minimum exposure if selected: w_{i,t} ≥ min_weight_if_selected * y_{i,t}
for t in periods_list:
    for i in assets_list:
        model.addConstr(
            w[i, t] >= min_weight_if_selected * y[i, t],
            name=f"min_exposure_{i}_{t}"
        )

# (14) Upper link between weight and selection (stocks): w_{i,t} ≤ max_weight_per_stock_asset * y_{i,t}
for t in periods_list:
    for i in stock_assets:
        model.addConstr(
            w[i, t] <= max_weight_per_stock_asset * y[i, t],
            name=f"stock_link_ub_{i}_{t}"
        )

# (15) Upper link between weight and selection (bonds): w_{i,t} ≤ max_weight_per_bond_asset * y_{i,t}
for t in periods_list:
    for i in bond_assets:
        model.addConstr(
            w[i, t] <= max_weight_per_bond_asset * y[i, t],
            name=f"bond_link_ub_{i}_{t}"
        )

# (16) Binary nature of y and (17) nonnegativity of w are already enforced by vtype and lb

# -----------------------------
# 6. Solve the model
# -----------------------------
model.optimize()

# -----------------------------
# 7. Print results and final answer
# -----------------------------
if model.Status == GRB.OPTIMAL:
    total_expected_return = model.ObjVal  # sum_t sum_i r_{i,t} * w_{i,t}
    # Expected multiple (decimal, not percent):
    # M = 1 + (1 / (12 * 100)) * Σ_t Σ_i r_{i,t}^{%} * w_{i,t}
    expected_multiple = 1.0 + total_expected_return / (num_periods * 100.0)

    print("Optimal total expected return over 12 periods (% of one-period capital):",
          total_expected_return)
    print("Maximum expected multiple (decimal):", expected_multiple)
    print("\nWeights per period (w_{i,t} > 1e-6):")
    for t in periods_list:
        print(f"Period {t}:")
        for i in assets_list:
            val = w[i, t].X
            if val > 1e-6:
                print(f"  {i:8s}: {val:.4f}")
        print()

    print("Number of assets invested per period (n_t):")
    for t in periods_list:
        print(f"  Period {t}: {int(round(n[t].X))}")

    # Required final output line
    print(f"FinalAnswer=【{expected_multiple}】")
else:
    print("Model did not reach an optimal solution. Status code:", model.Status)
    # If no optimal solution, still output something for FinalAnswer as per format
    print("FinalAnswer=【None】")