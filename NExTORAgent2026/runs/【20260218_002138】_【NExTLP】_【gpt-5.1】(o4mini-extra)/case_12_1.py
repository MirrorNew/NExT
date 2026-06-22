import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters and Data
# =========================

# From Parameters List (must use exactly these names/values)
NumberOfPeriods = 12
AssetsPerPeriod = 6
OverallRiskLimit = 6
MaxSingleStockProportion = 0.4
MaxSingleBondProportion = 0.6
MaxTotalStockProportion = 0.7
MinTotalBondProportion = 0.3
MinAssetsSelected = 5
AvgWeight_ZM_Limit = 0.4
MinExposure = 0.1

ExpectedAnnualReturnPercent = {
    'Stock X': 12,
    'Stock Y': 10,
    'Stock Z': 15,
    'Bond M': 5,
    'Bond N': 4,
    'Bond O': 6
}

RiskIndex = {
    'Stock X': 8,
    'Stock Y': 6,
    'Stock Z': 10,
    'Bond M': 2,
    'Bond N': 1,
    'Bond O': 3
}

CurrentExpectedReturn = [
    None,
    {'X': 11.5, 'Y': 9.8, 'Z': 14.2, 'M': 5.1, 'N': 3.9, 'O': 6.2},
    {'X': 12.3, 'Y': 10.2, 'Z': 15.1, 'M': 4.9, 'N': 4.1, 'O': 5.8},
    {'X': 11.8, 'Y': 9.5, 'Z': 14.7, 'M': 5.0, 'N': 4.0, 'O': 6.0},
    {'X': 12.0, 'Y': 10.0, 'Z': 15.3, 'M': 5.2, 'N': 3.8, 'O': 6.1},
    {'X': 12.1, 'Y': 10.1, 'Z': None, 'M': 5.0, 'N': 4.2, 'O': 5.9},
    {'X': 11.9, 'Y': 9.7, 'Z': 15.0, 'M': 5.3, 'N': 4.0, 'O': 6.3},
    {'X': 12.4, 'Y': 10.3, 'Z': 15.2, 'M': 5.1, 'N': 4.1, 'O': 6.0},
    {'X': 12.2, 'Y': 10.0, 'Z': 14.9, 'M': 5.0, 'N': 3.9, 'O': 6.2},
    {'X': 11.7, 'Y': 9.6, 'Z': 14.5, 'M': 4.8, 'N': 4.0, 'O': 6.1},
    {'X': 12.5, 'Y': 10.4, 'Z': None, 'M': 5.2, 'N': 4.3, 'O': 5.7},
    {'X': 12.0, 'Y': 10.1, 'Z': 15.4, 'M': 5.1, 'N': 4.2, 'O': 6.0},
    {'X': 11.6, 'Y': 9.9, 'Z': 14.8, 'M': 5.0, 'N': 4.1, 'O': 6.2}
]

# Basic sets
periods = range(1, NumberOfPeriods + 1)
assets = ['X', 'Y', 'Z', 'M', 'N', 'O']
stocks = ['X', 'Y', 'Z']
bonds = ['M', 'N', 'O']

# Map short asset keys to the risk indices in RiskIndex
risk_index = {
    'X': RiskIndex['Stock X'],
    'Y': RiskIndex['Stock Y'],
    'Z': RiskIndex['Stock Z'],
    'M': RiskIndex['Bond M'],
    'N': RiskIndex['Bond N'],
    'O': RiskIndex['Bond O']
}

# Returns per period and asset converted to decimal
ret = {}
for t in periods:
    ret[t] = {}
    for a in assets:
        r_pct = CurrentExpectedReturn[t][a]
        if r_pct is None:
            ret[t][a] = 0.0
        else:
            ret[t][a] = r_pct / 100.0

# Availability: 1 if investable, 0 if not; based strictly on CurrentExpectedReturn being None
avail = {}
for t in periods:
    avail[t] = {}
    for a in assets:
        avail[t][a] = 0 if CurrentExpectedReturn[t][a] is None else 1

# =========================
# 2. Create Model
# =========================

model = gp.Model("Dynamic_MultiPeriod_Asset_Allocation")

# =========================
# 3. Decision Variables
# =========================

# w[t,a] = proportion of total funds in period t allocated to asset a
w = model.addVars(periods, assets, name="w", lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)

# y[t,a] = 1 if asset a is selected in period t, 0 otherwise
y = model.addVars(periods, assets, name="y", vtype=GRB.BINARY)

# =========================
# 4. Objective Function
# =========================

# Maximize total expected return multiple over 12 periods
model.setObjective(
    gp.quicksum(w[t, a] * ret[t][a] for t in periods for a in assets),
    GRB.MAXIMIZE
)

# =========================
# 5. Constraints
# =========================

# (1) Full investment: sum of weights = 1 in each period
for t in periods:
    model.addConstr(gp.quicksum(w[t, a] for a in assets) == 1.0, name=f"FullInvestment_t{t}")

# (2) Non-investable assets (weights) and (3) Non-investable assets (selection)
for t in periods:
    for a in assets:
        model.addConstr(w[t, a] <= avail[t][a], name=f"AvailW_t{t}_{a}")
        model.addConstr(y[t, a] <= avail[t][a], name=f"AvailY_t{t}_{a}")

# (4) Risk limit per period: sum w[t,a] * risk[a] <= OverallRiskLimit
for t in periods:
    model.addConstr(
        gp.quicksum(w[t, a] * risk_index[a] for a in assets) <= OverallRiskLimit,
        name=f"RiskLimit_t{t}"
    )

# (5) Single stock upper bound per period per stock asset
for t in periods:
    for a in stocks:
        model.addConstr(
            w[t, a] <= MaxSingleStockProportion,
            name=f"MaxSingleStock_t{t}_{a}"
        )

# (6) Single bond upper bound per period per bond asset
for t in periods:
    for a in bonds:
        model.addConstr(
            w[t, a] <= MaxSingleBondProportion,
            name=f"MaxSingleBond_t{t}_{a}"
        )

# (7) Total stock proportion cap per period
for t in periods:
    model.addConstr(
        gp.quicksum(w[t, a] for a in stocks) <= MaxTotalStockProportion,
        name=f"TotalStockCap_t{t}"
    )

# (8) Total bond proportion floor per period
for t in periods:
    model.addConstr(
        gp.quicksum(w[t, a] for a in bonds) >= MinTotalBondProportion,
        name=f"TotalBondFloor_t{t}"
    )

# (9) Minimum number of assets selected per period
for t in periods:
    model.addConstr(
        gp.quicksum(y[t, a] for a in assets) >= MinAssetsSelected,
        name=f"MinAssetsSelected_t{t}"
    )

# (10) Policy limit on Z and M total weights: sum_t (w[t,Z] + w[t,M]) <= AvgWeight_ZM_Limit * NumberOfPeriods
model.addConstr(
    gp.quicksum(w[t, 'Z'] + w[t, 'M'] for t in periods) <= AvgWeight_ZM_Limit * NumberOfPeriods,
    name="PolicyLimit_ZM"
)

# (11) Minimum exposure when selected: w[t,a] >= MinExposure * y[t,a]
for t in periods:
    for a in assets:
        model.addConstr(
            w[t, a] >= MinExposure * y[t, a],
            name=f"MinExposure_t{t}_{a}"
        )

# (12) Linking w and y (upper): w[t,a] <= y[t,a]
for t in periods:
    for a in assets:
        model.addConstr(
            w[t, a] <= y[t, a],
            name=f"LinkWY_t{t}_{a}"
        )

# =========================
# 6. Optimize
# =========================

model.Params.OutputFlag = 0  # turn off solver output; set to 1 to see logs
model.optimize()

# =========================
# 7. Results
# =========================

if model.SolCount > 0:
    print("Optimal objective (total expected return multiple over 12 periods):", model.ObjVal)
    print("\nOptimal allocation w[t,a]:")
    for t in periods:
        print(f"Period {t}:")
        for a in assets:
            w_val = w[t, a].X
            y_val = y[t, a].X
            if w_val > 1e-6 or y_val > 0.5:
                print(f"  Asset {a}: weight={w_val:.4f}, selected={int(round(y_val))}")
else:
    print("No feasible solution found.")

# FinalAnswer is the maximum expected multiple (objective value), decimal (not percentage)
final_answer_value = model.ObjVal if model.SolCount > 0 else float('nan')
print(f"FinalAnswer=【{final_answer_value}】")