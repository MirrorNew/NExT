import gurobipy as gp
from gurobipy import GRB

# ====================
# 1. PARAMETERS
# ====================
portfolio_worth = 1000000
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

# Asset data from Table_1_asset_data
# ['Stock S', 0.06, 100000, 400000, 0.5, 0.3],
# ['Real Estate R', 0.07, 300000, 1000000, 0.7, 0.25],
# ['Bond B', 0.05, 100000, 1000000, 0.8, 0.1],
# ['Certificate of Deposit C', 0.04, 100000, 300000, 0.9, 0.05],
# Note: Hedging Product D exists in the table but is not used in the model (from context)

# Asset indices for easy access
assets = ['S', 'R', 'B', 'C']
return_rates = {'S': 0.06, 'R': 0.07, 'B': 0.05, 'C': 0.04}
min_invest = {'S': 100000, 'R': 300000, 'B': 100000, 'C': 100000}
max_invest = {'S': 400000, 'R': 1000000, 'B': 1000000, 'C': 300000}
esg_scores = {'S': 0.5, 'R': 0.7, 'B': 0.8, 'C': 0.9}
risk_factors = {'S': 0.3, 'R': 0.25, 'B': 0.1, 'C': 0.05}

# Big M for linearization (large enough but not too large to cause numerical issues)
M = 1e7

# ====================
# 2. CREATE MODEL
# ====================
model = gp.Model("AssetAllocation")

# ====================
# 3. DECISION VARIABLES
# ====================
# Investment amounts (USD)
x_S = model.addVar(lb=min_invest['S'], ub=max_invest['S'], vtype=GRB.CONTINUOUS, name="x_S")
x_R = model.addVar(lb=min_invest['R'], ub=max_invest['R'], vtype=GRB.CONTINUOUS, name="x_R")
x_B = model.addVar(lb=min_invest['B'], ub=max_invest['B'], vtype=GRB.CONTINUOUS, name="x_B")
x_C = model.addVar(lb=min_invest['C'], ub=max_invest['C'], vtype=GRB.CONTINUOUS, name="x_C")

# Binary indicators for at least $500,000 investment
y_S = model.addVar(vtype=GRB.BINARY, name="y_S")
y_R = model.addVar(vtype=GRB.BINARY, name="y_R")
y_B = model.addVar(vtype=GRB.BINARY, name="y_B")
y_C = model.addVar(vtype=GRB.BINARY, name="y_C")

# Binary indicator for risk reserve (z=1 if weighted risk > 0.2)
z = model.addVar(vtype=GRB.BINARY, name="z")

# ====================
# 4. OBJECTIVE FUNCTION
# ====================
# Maximize total annualized return
total_return = 0.06*x_S + 0.07*x_R + 0.05*x_B + 0.04*x_C
model.setObjective(total_return, GRB.MAXIMIZE)

# ====================
# 5. CONSTRAINTS
# ====================
# 5.1 Full investment (all funds must be invested)
model.addConstr(x_S + x_R + x_B + x_C == portfolio_worth, name="full_investment")

# 5.2 Pairwise investment limit (any two assets ≤ 700,000)
model.addConstr(x_S + x_R <= sum_of_any_two_investments_max, name="pair_S_R")
model.addConstr(x_S + x_B <= sum_of_any_two_investments_max, name="pair_S_B")
model.addConstr(x_S + x_C <= sum_of_any_two_investments_max, name="pair_S_C")
model.addConstr(x_R + x_B <= sum_of_any_two_investments_max, name="pair_R_B")
model.addConstr(x_R + x_C <= sum_of_any_two_investments_max, name="pair_R_C")
model.addConstr(x_B + x_C <= sum_of_any_two_investments_max, name="pair_B_C")

# 5.3 Liquidity lower bound (Bonds + Time Deposits ≥ 200,000)
model.addConstr(x_B + x_C >= time_deposits_plus_bonds_min, name="liquidity_min")

# 5.4 Time deposit upper bound (already in variable bounds, but explicit for clarity)
model.addConstr(x_C <= time_deposits_max, name="time_deposit_max")

# 5.5 Real estate minimum proportion (≥30% of total)
model.addConstr(x_R >= real_estate_proportion_min * (x_S + x_R + x_B + x_C), name="real_estate_min_prop")

# 5.6 Stock upper bound (risk control - already in variable bounds)
model.addConstr(x_S <= stock_investment_max, name="stock_max")

# 5.7 Bond minimum (already in variable bounds, but explicit)
model.addConstr(x_B >= bond_investment_min, name="bond_min")

# 5.8 Stock to (Real Estate + Bond) ratio
model.addConstr(x_S <= stock_to_others_ratio_max * (x_R + x_B), name="stock_to_RE_Bond_ratio")

# 5.9 Diversification: at least one asset ≥ 500,000
# Indicator constraints using addGenConstrIndicator
model.addGenConstrIndicator(y_S, 1, x_S >= at_least_one_asset_investment_min, name="indicator_y_S_1")
model.addGenConstrIndicator(y_S, 0, x_S <= at_least_one_asset_investment_min - 1, name="indicator_y_S_0")

model.addGenConstrIndicator(y_R, 1, x_R >= at_least_one_asset_investment_min, name="indicator_y_R_1")
model.addGenConstrIndicator(y_R, 0, x_R <= at_least_one_asset_investment_min - 1, name="indicator_y_R_0")

model.addGenConstrIndicator(y_B, 1, x_B >= at_least_one_asset_investment_min, name="indicator_y_B_1")
model.addGenConstrIndicator(y_B, 0, x_B <= at_least_one_asset_investment_min - 1, name="indicator_y_B_0")

model.addGenConstrIndicator(y_C, 1, x_C >= at_least_one_asset_investment_min, name="indicator_y_C_1")
model.addGenConstrIndicator(y_C, 0, x_C <= at_least_one_asset_investment_min - 1, name="indicator_y_C_0")

# At least one y_i must be 1
model.addConstr(y_S + y_R + y_B + y_C >= 1, name="diversification_at_least_one")

# 5.10 ESG weighted average constraint
# 0.5*x_S + 0.7*x_R + 0.8*x_B + 0.9*x_C ≥ 0.7*(x_S + x_R + x_B + x_C)
model.addConstr(
    esg_scores['S']*x_S + esg_scores['R']*x_R + esg_scores['B']*x_B + esg_scores['C']*x_C
    >= weighted_ESG_score_min * (x_S + x_R + x_B + x_C),
    name="ESG_weighted_avg"
)

# 5.11 Risk-reserve disjunction
# If weighted average risk > 0.2 (z=1), then risk constraint can be violated but portfolio size reduced by 200,000
# Weighted risk: 0.3*x_S + 0.25*x_R + 0.1*x_B + 0.05*x_C
weighted_risk = risk_factors['S']*x_S + risk_factors['R']*x_R + risk_factors['B']*x_B + risk_factors['C']*x_C

# Constraint 1: weighted_risk ≤ 0.2*total + M*z
# When z=0: weighted_risk ≤ 0.2*total (strict risk constraint)
# When z=1: weighted_risk ≤ 0.2*total + M (constraint relaxed via big M)
model.addConstr(
    weighted_risk <= weighted_risk_factor_max * (x_S + x_R + x_B + x_C) + M * z,
    name="risk_constraint_with_bigM"
)

# Constraint 2: total investment ≤ 1,000,000 - 200,000*z
# When z=0: total ≤ 1,000,000 (normal)
# When z=1: total ≤ 800,000 (reserve applies)
model.addConstr(
    x_S + x_R + x_B + x_C <= portfolio_worth - additional_risk_reserve * z,
    name="risk_reserve_constraint"
)

# ====================
# 6. SOLVE
# ====================
model.optimize()

# ====================
# 7. OUTPUT RESULTS
# ====================
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"Objective (Total Annual Return): ${model.objVal:.2f}")
    print("\nInvestment amounts:")
    print(f"  Stocks (S): ${x_S.X:,.2f}")
    print(f"  Real Estate (R): ${x_R.X:,.2f}")
    print(f"  Bonds (B): ${x_B.X:,.2f}")
    print(f"  Time Deposits (C): ${x_C.X:,.2f}")
    
    print("\nDiversification indicators (1 if investment ≥ $500,000):")
    print(f"  y_S: {y_S.X}")
    print(f"  y_R: {y_R.X}")
    print(f"  y_B: {y_B.X}")
    print(f"  y_C: {y_C.X}")
    
    print(f"\nRisk reserve indicator (z): {z.X} (1 means reserve applies)")
    
    total_inv = x_S.X + x_R.X + x_B.X + x_C.X
    weighted_risk_val = (risk_factors['S']*x_S.X + risk_factors['R']*x_R.X + 
                         risk_factors['B']*x_B.X + risk_factors['C']*x_C.X) / total_inv
    print(f"\nWeighted average risk factor: {weighted_risk_val:.4f}")
    
    # The question asks for the maximum annualized return
    print(f"FinalAnswer=【{model.objVal:.2f}】")
else:
    print("No optimal solution found.")
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible.")
    elif model.status == GRB.UNBOUNDED:
        print("Model is unbounded.")
    else:
        print(f"Optimization ended with status: {model.status}")
    print(f"FinalAnswer=【None】")