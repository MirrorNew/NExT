import gurobipy as gp
from gurobipy import GRB

# ==============================
# 1. Parameters (from Parameters List)
# ==============================
portfolio_worth = 1000000
number_of_asset_types = 4
total_investment_max = 1000000
sum_of_any_two_investments_max = 700000
time_deposits_plus_bonds_min = 200000
time_deposits_max = 300000
real_estate_proportion_min = 0.3
stock_investment_max = 400000
bond_investment_min = 100000
stock_to_others_ratio_max = 0.5
at_least_one_asset_investment_min = 500000
weighted_ESG_score_min = 0.7
weighted_risk_factor_max = 0.2
additional_risk_reserve = 200000
Table_1_asset_data = [
    ['Stock S', 0.06, 100000, 400000, 0.5, 0.3],
    ['Real Estate R', 0.07, 300000, 1000000, 0.7, 0.25],
    ['Bond B', 0.05, 100000, 1000000, 0.8, 0.1],
    ['Certificate of Deposit C', 0.04, 100000, 300000, 0.9, 0.05],
    ['Hedging Product D', 0.1, 500000, 1000000, 0.3, 0.8]
]

# Extract useful parameters from the table for S, R, B, C
# Indices: 0-Stock S, 1-Real Estate R, 2-Bond B, 3-CD C
returns = {
    'S': Table_1_asset_data[0][1],
    'R': Table_1_asset_data[1][1],
    'B': Table_1_asset_data[2][1],
    'C': Table_1_asset_data[3][1],
}
min_invest = {
    'S': Table_1_asset_data[0][2],
    'R': Table_1_asset_data[1][2],
    'B': Table_1_asset_data[2][2],
    'C': Table_1_asset_data[3][2],
}
max_invest = {
    'S': Table_1_asset_data[0][3],
    'R': Table_1_asset_data[1][3],
    'B': Table_1_asset_data[2][3],
    'C': Table_1_asset_data[3][3],
}
esg = {
    'S': Table_1_asset_data[0][4],
    'R': Table_1_asset_data[1][4],
    'B': Table_1_asset_data[2][4],
    'C': Table_1_asset_data[3][4],
}
risk = {
    'S': Table_1_asset_data[0][5],
    'R': Table_1_asset_data[1][5],
    'B': Table_1_asset_data[2][5],
    'C': Table_1_asset_data[3][5],
}

assets = ['S', 'R', 'B', 'C']

# ==============================
# 2. Create model
# ==============================
model = gp.Model("XYZ_Asset_Allocation")

# ==============================
# 3. Decision variables
# ==============================
# Investment amounts
x = model.addVars(assets, vtype=GRB.CONTINUOUS, name="x")

# Diversification indicators: y_i = 1 if x_i >= 500000
y = model.addVars(assets, vtype=GRB.BINARY, name="y")

# Risk-reserve indicator: z = 1 if weighted average risk > 0.2 (reserve required)
z = model.addVar(vtype=GRB.BINARY, name="z")

# ==============================
# 4. Objective: Maximize annualized return
# ==============================
obj = (
    returns['S'] * x['S']
    + returns['R'] * x['R']
    + returns['B'] * x['B']
    + returns['C'] * x['C']
)
model.setObjective(obj, GRB.MAXIMIZE)

# ==============================
# 5. Constraints
# ==============================

# Total investment variable (for convenience)
total_invest = x['S'] + x['R'] + x['B'] + x['C']

# 5.1 Full investment and budget upper bound
model.addConstr(total_invest == portfolio_worth, name="FullInvestment")
model.addConstr(total_invest <= total_investment_max, name="BudgetUpperBound")

# 5.2 Pairwise investment limits (any two <= 700000)
model.addConstr(x['S'] + x['R'] <= sum_of_any_two_investments_max, name="Pair_SR")
model.addConstr(x['S'] + x['B'] <= sum_of_any_two_investments_max, name="Pair_SB")
model.addConstr(x['S'] + x['C'] <= sum_of_any_two_investments_max, name="Pair_SC")
model.addConstr(x['R'] + x['B'] <= sum_of_any_two_investments_max, name="Pair_RB")
model.addConstr(x['R'] + x['C'] <= sum_of_any_two_investments_max, name="Pair_RC")
model.addConstr(x['B'] + x['C'] <= sum_of_any_two_investments_max, name="Pair_BC")

# 5.3 Liquidity constraints
model.addConstr(x['B'] + x['C'] >= time_deposits_plus_bonds_min, name="Liquidity_BplusC")
model.addConstr(x['C'] <= time_deposits_max, name="CD_Max_Parameter")

# 5.4 Real estate minimum proportion
model.addConstr(
    x['R'] >= real_estate_proportion_min * total_invest,
    name="RealEstate_MinProp",
)

# 5.5 Stock and bond bounds (from narrative and table)
model.addConstr(x['S'] <= stock_investment_max, name="Stock_Max_Parameter")
model.addConstr(x['B'] >= bond_investment_min, name="Bond_Min_Parameter")

# 5.6 Stock-to-(Real Estate + Bond) ratio
model.addConstr(
    x['S'] <= stock_to_others_ratio_max * (x['R'] + x['B']),
    name="Stock_to_REplusBond",
)

# 5.7 Diversification OR: at least one asset >= 500000
# Use indicator constraints x_i >= 500000 iff y_i = 1 (only enforcing the => direction, per spec)
at_least_one_min = at_least_one_asset_investment_min

# Indicator constraints: y_i = 1 -> x_i >= 500000
for a in assets:
    model.addGenConstrIndicator(
        y[a], 1, x[a] >= at_least_one_min, name=f"Ind_{a}_ge_{at_least_one_min}"
    )

# At least one asset satisfies that (i.e., some y_i = 1)
model.addConstr(gp.quicksum(y[a] for a in assets) >= 1, name="Diversification_OR")

# 5.8 ESG weighted average >= 0.7
model.addConstr(
    esg['S'] * x['S'] + esg['R'] * x['R'] + esg['B'] * x['B'] + esg['C'] * x['C']
    >= weighted_ESG_score_min * total_invest,
    name="ESG_Weighted_Avg",
)

# 5.9 Risk-weighted average and risk-reserve disjunction
# Unnormalized risk
R_expr = (
    risk['S'] * x['S']
    + risk['R'] * x['R']
    + risk['B'] * x['B']
    + risk['C'] * x['C']
)

# The original model description already included a big-M style disjunction:
# (0.3*x_S+0.25*x_R+0.1*x_B+0.05*x_C) ≤ 0.2*(x_S+x_R+x_B+x_C) + M*z
# x_S+x_R+x_B+x_C ≤ 1000000 - 200000*z
# z∈{0,1}
#
# We must NOT introduce our own M; we instead implement the same logical structure
# via indicator constraints without big-M, as required.

# Case z = 0: weighted average risk <= 0.2
model.addGenConstrIndicator(
    z, 0,
    R_expr <= weighted_risk_factor_max * total_invest,
    name="RiskBound_if_z0",
)

# Case z = 1: total invest <= 1000000 - 200000 (reserve taken out)
model.addGenConstrIndicator(
    z, 1,
    total_invest <= total_investment_max - additional_risk_reserve,
    name="Reserve_if_z1",
)

# 5.10 Asset-specific lower/upper bounds from table (reinforcing)
for a in assets:
    model.addConstr(x[a] >= min_invest[a], name=f"{a}_Min_Table")
    model.addConstr(x[a] <= max_invest[a], name=f"{a}_Max_Table")

# ==============================
# 6. Optimize
# ==============================
model.Params.OutputFlag = 0  # silence solver output; remove or set to 1 if you want logs
model.optimize()

# ==============================
# 7. Print results
# ==============================
if model.status == GRB.OPTIMAL:
    opt_obj = model.objVal
    print("Optimal annualized return:", opt_obj)
    for a in assets:
        print(f"x_{a} = {x[a].X}")
    for a in assets:
        print(f"y_{a} = {y[a].X}")
    print(f"z = {z.X}")
else:
    opt_obj = None
    print("No optimal solution found. Model status:", model.status)

# FinalAnswer is the maximum annualized return
print(f"FinalAnswer=【{opt_obj}】")