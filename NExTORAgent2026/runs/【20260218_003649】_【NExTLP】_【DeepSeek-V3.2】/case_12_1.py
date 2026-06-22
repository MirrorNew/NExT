import gurobipy as gp
from gurobipy import GRB

# ==================== Parameters ====================
# From Parameters List
T = 12  # NumberOfPeriods
assets = ['X', 'Y', 'Z', 'M', 'N', 'O']  # Assets
stocks = ['X', 'Y', 'Z']
bonds = ['M', 'N', 'O']

# Risk index from Table 1
risk_index = {
    'X': 8, 'Y': 6, 'Z': 10,
    'M': 2, 'N': 1, 'O': 3
}

# Current expected returns from Table 2 (periods 1-12)
# The Parameters List has period 0 as None, so we skip index 0
return_data = [
    None,  # index 0 not used
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

# Create availability matrix (1 if investable, 0 if not)
avail = {}
for t in range(1, T+1):
    for i in assets:
        if return_data[t][i] is None:
            avail[t, i] = 0
        else:
            avail[t, i] = 1

# Convert returns to decimal (divide by 100)
returns = {}
for t in range(1, T+1):
    for i in assets:
        if avail[t, i] == 1:
            returns[t, i] = return_data[t][i] / 100.0
        else:
            returns[t, i] = 0.0  # Not used due to availability constraints

# Other parameters from Parameters List
overall_risk_limit = 6
max_single_stock = 0.4
max_single_bond = 0.6
max_total_stock = 0.7
min_total_bond = 0.3
min_assets_selected = 5  # Note: Problem says "more than 4", so ≥5
avg_weight_ZM_limit = 0.4
min_exposure = 0.1

# ==================== Model ====================
model = gp.Model("InvestmentPortfolioOptimization")

# ==================== Variables ====================
w = {}  # Proportion variables
y = {}  # Binary selection variables

for t in range(1, T+1):
    for i in assets:
        w[t, i] = model.addVar(lb=0.0, ub=1.0, name=f"w_{t}_{i}")
        y[t, i] = model.addVar(vtype=GRB.BINARY, name=f"y_{t}_{i}")

# ==================== Objective ====================
obj_expr = gp.quicksum(returns[t, i] * w[t, i] for t in range(1, T+1) for i in assets)
model.setObjective(obj_expr, GRB.MAXIMIZE)

# ==================== Constraints ====================
# 1. Full investment each period
for t in range(1, T+1):
    model.addConstr(gp.quicksum(w[t, i] for i in assets) == 1, name=f"FullInvestment_t{t}")

# 2. Non-investable assets constraints
for t in range(1, T+1):
    for i in assets:
        model.addConstr(w[t, i] <= avail[t, i], name=f"Avail_w_{t}_{i}")
        model.addConstr(y[t, i] <= avail[t, i], name=f"Avail_y_{t}_{i}")

# 3. Risk limit per period
for t in range(1, T+1):
    model.addConstr(
        gp.quicksum(risk_index[i] * w[t, i] for i in assets) <= overall_risk_limit,
        name=f"RiskLimit_t{t}"
    )

# 4. Single asset upper bounds
for t in range(1, T+1):
    for i in stocks:
        model.addConstr(w[t, i] <= max_single_stock, name=f"MaxStock_{t}_{i}")
    for i in bonds:
        model.addConstr(w[t, i] <= max_single_bond, name=f"MaxBond_{t}_{i}")

# 5. Diversification constraints
for t in range(1, T+1):
    # Total stock proportion cap
    model.addConstr(
        gp.quicksum(w[t, i] for i in stocks) <= max_total_stock,
        name=f"MaxTotalStock_t{t}"
    )
    # Total bond proportion floor
    model.addConstr(
        gp.quicksum(w[t, i] for i in bonds) >= min_total_bond,
        name=f"MinTotalBond_t{t}"
    )
    # Minimum number of assets selected per period
    model.addConstr(
        gp.quicksum(y[t, i] for i in assets) >= min_assets_selected,
        name=f"MinAssets_t{t}"
    )

# 6. Policy limit on Z and M (average over periods)
model.addConstr(
    gp.quicksum(w[t, 'Z'] + w[t, 'M'] for t in range(1, T+1)) <= avg_weight_ZM_limit * T,
    name="PolicyLimit_ZM"
)

# 7. Linking constraints between w and y
for t in range(1, T+1):
    for i in assets:
        # Minimum exposure when selected
        model.addConstr(w[t, i] >= min_exposure * y[t, i], name=f"MinExposure_{t}_{i}")
        # Upper bound linking (if not selected, w must be 0)
        model.addConstr(w[t, i] <= y[t, i], name=f"Linking_{t}_{i}")

# ==================== Solve ====================
model.optimize()

# ==================== Results ====================
if model.status == GRB.OPTIMAL:
    total_return = model.objVal
    print(f"Optimal total expected return multiple: {total_return:.6f}")
    
    # Print detailed allocation for each period (optional)
    for t in range(1, T+1):
        print(f"\nPeriod {t}:")
        selected_assets = [i for i in assets if y[t, i].X > 0.5]
        print(f"  Selected assets: {selected_assets}")
        for i in assets:
            if w[t, i].X > 1e-6:
                print(f"    {i}: {w[t, i].X:.4f} (return: {returns[t, i]:.4f})")
    
    print(f"\nFinalAnswer=【{total_return:.6f}】")
else:
    print("No optimal solution found.")
    print(f"FinalAnswer=【0】")