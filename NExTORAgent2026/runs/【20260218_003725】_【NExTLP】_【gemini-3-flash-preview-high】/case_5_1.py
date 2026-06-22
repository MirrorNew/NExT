import gurobipy as gp
from gurobipy import GRB

# XYZ Asset Management Portfolio Optimization
# Assets: Stocks (S), Real Estate (R), Bonds (B), Certificate of Deposit (C)

# 1. Create the model
model = gp.Model("XYZ_Asset_Management")

# 2. Parameters from the problem description
portfolio_worth = 1000000
additional_risk_reserve = 200000
sum_max_any_two = 700000
time_deposits_plus_bonds_min = 200000
time_deposits_max = 300000
real_estate_proportion_min = 0.3
stock_investment_max_val = 400000
bond_investment_min_val = 100000
stock_to_others_ratio_max = 0.5
diversification_min = 500000
weighted_ESG_min = 0.7
weighted_risk_max = 0.2

# Asset Specific Data
# [Expected Return, Min Investment, Max Investment, ESG Score, Risk Factor]
assets = {
    'S': {'ret': 0.06, 'min': 100000, 'max': 400000, 'esg': 0.5, 'risk': 0.30},
    'R': {'ret': 0.07, 'min': 300000, 'max': 1000000, 'esg': 0.7, 'risk': 0.25},
    'B': {'ret': 0.05, 'min': 100000, 'max': 1000000, 'esg': 0.8, 'risk': 0.10},
    'C': {'ret': 0.04, 'min': 100000, 'max': 300000, 'esg': 0.9, 'risk': 0.05}
}

# 3. Decision Variables
# Continuous variables for investment amounts
x = model.addVars(assets.keys(), lb=0, name="x")
for a in assets:
    x[a].lb = assets[a]['min']
    x[a].ub = assets[a]['max']

# Binary variables for compliance indicators
y = model.addVars(assets.keys(), vtype=GRB.BINARY, name="y")
# Indicator for risk management (z=1 if weighted risk > 0.2)
z = model.addVar(vtype=GRB.BINARY, name="z")

# 4. Objective Function
# Maximize total annualized return
model.setObjective(gp.quicksum(assets[a]['ret'] * x[a] for a in assets), GRB.MAXIMIZE)

# 5. Constraints
# Total investment amount dependent on risk management threshold
T = gp.quicksum(x[a] for a in assets)
model.addConstr(T == portfolio_worth - additional_risk_reserve * z)

# Pairwise investment limits: sum of any two investments <= $700,000
asset_list = list(assets.keys())
for i in range(len(asset_list)):
    for j in range(i + 1, len(asset_list)):
        model.addConstr(x[asset_list[i]] + x[asset_list[j]] <= sum_max_any_two)

# Liquidity requirements
model.addConstr(x['B'] + x['C'] >= time_deposits_plus_bonds_min)
model.addConstr(x['C'] <= time_deposits_max)

# Real estate minimum proportion
model.addConstr(x['R'] >= real_estate_proportion_min * T)

# Stocks and bond constraints
model.addConstr(x['S'] <= stock_investment_max_val)
model.addConstr(x['B'] >= bond_investment_min_val)
model.addConstr(x['S'] <= stock_to_others_ratio_max * (x['R'] + x['B']))

# Weighted ESG score constraint
model.addConstr(gp.quicksum(assets[a]['esg'] * x[a] for a in assets) >= weighted_ESG_min * T)

# Asset Diversification Indicator Constraints
# y[a] = 1 if x[a] >= $500,000, else 0
for a in assets:
    model.addGenConstrIndicator(y[a], 1, x[a] >= diversification_min)
    model.addGenConstrIndicator(y[a], 0, x[a] <= diversification_min - 0.001)

# At least one asset category must exceed $500,000
model.addConstr(gp.quicksum(y[a] for a in assets) >= 1)

# Risk Reserve Management Indicator Constraints
# z = 1 if weighted average risk factor > 0.2, else 0
risk_sum = gp.quicksum(assets[a]['risk'] * x[a] for a in assets)
model.addGenConstrIndicator(z, 0, risk_sum <= weighted_risk_max * T)
model.addGenConstrIndicator(z, 1, risk_sum >= weighted_risk_max * T + 0.0001)

# 6. Solve and print the results
model.optimize()

if model.status == GRB.OPTIMAL:
    max_annualized_return = model.objVal
    print(f"FinalAnswer=【{max_annualized_return}】")
else:
    print("Optimal solution was not found.")