import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs
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

# Table data: [Name, Expected Annual Return, Min, Max, ESG Score, Risk Factor]
# We map these to a dictionary for the 4 active assets (S, R, B, C). 
# Hedging Product D is excluded based on the problem statement ("not implemented").
assets = ['S', 'R', 'B', 'C']
asset_data = {
    'S': {'ret': 0.06, 'min': 100000, 'max': 400000, 'esg': 0.5, 'risk': 0.30},
    'R': {'ret': 0.07, 'min': 300000, 'max': 1000000, 'esg': 0.7, 'risk': 0.25},
    'B': {'ret': 0.05, 'min': 100000, 'max': 1000000, 'esg': 0.8, 'risk': 0.10},
    'C': {'ret': 0.04, 'min': 100000, 'max': 300000, 'esg': 0.9, 'risk': 0.05}
}

# 2. Create the model
model = gp.Model("XYZ_Asset_Management_Portfolio")

# 3. Create decision variables
x = {} # Continuous variables for investment amounts
y = {} # Binary variables for diversification indicator (x >= 500,000)

for i in assets:
    x[i] = model.addVar(lb=asset_data[i]['min'], ub=asset_data[i]['max'], vtype=GRB.CONTINUOUS, name=f"x_{i}")
    y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")

# Binary variable for Risk Reserve trigger (z=1 if reserve paid, z=0 if no reserve)
z = model.addVar(vtype=GRB.BINARY, name="z")

# 4. Set up the objective function
# Maximize total annualized return
obj_expr = gp.quicksum(asset_data[i]['ret'] * x[i] for i in assets)
model.setObjective(obj_expr, GRB.MAXIMIZE)

# 5. Add Constraints

# Helper expression for total investment amount in the portfolio
total_investment = gp.quicksum(x[i] for i in assets)

# Constraint: Budget & Full Investment
# If z=0 (Low Risk), invest 1,000,000. If z=1 (High Risk), invest 1,000,000 - 200,000 = 800,000.
model.addConstr(total_investment == portfolio_worth - additional_risk_reserve * z, "Budget_Full_Investment")

# Constraint: Pairwise Investment Limit
# The sum of any two investment amounts cannot exceed 700,000
import itertools
for i, j in itertools.combinations(assets, 2):
    model.addConstr(x[i] + x[j] <= sum_of_any_two_investments_max, f"Pairwise_Max_{i}_{j}")

# Constraint: Liquidity (Bonds + Time Deposits >= 200,000)
model.addConstr(x['B'] + x['C'] >= time_deposits_plus_bonds_min, "Liquidity_Requirement")

# Constraint: Time Deposit Upper Bound (Explicitly added, though covered by variable bounds)
model.addConstr(x['C'] <= time_deposits_max, "Time_Deposit_Max_Constraint")

# Constraint: Real Estate Proportion (>= 30% of total investment)
model.addConstr(x['R'] >= real_estate_proportion_min * total_investment, "Real_Estate_Min_Ratio")

# Constraint: Stock to Other Assets Ratio
# Stock <= 0.5 * (Real Estate + Bond)
model.addConstr(x['S'] <= stock_to_others_ratio_max * (x['R'] + x['B']), "Stock_Ratio_Limit")

# Constraint: Diversification
# At least one asset category must exceed 500,000. 
# We use binary variables y[i]. If y[i]=1, x[i] must be >= 500,000.
# Sum of y must be >= 1.
model.addConstr(gp.quicksum(y[i] for i in assets) >= 1, "Diversification_Trigger_Count")

for i in assets:
    # Indicator constraint: y[i] = 1 implies x[i] >= 500,000
    model.addGenConstrIndicator(y[i], 1, x[i] >= at_least_one_asset_investment_min, name=f"Div_Indicator_{i}")

# Constraint: ESG Weighted Average
# Weighted average ESG score >= 0.7
# sum(score * x) >= 0.7 * total_investment
esg_lhs = gp.quicksum(asset_data[i]['esg'] * x[i] for i in assets)
model.addConstr(esg_lhs >= weighted_ESG_score_min * total_investment, "ESG_Requirement")

# Constraint: Risk Weighted Average
# Weighted average risk factor <= 0.2.
# Logic: If weighted risk <= 0.2, we don't need reserve (z can be 0).
# If weighted risk > 0.2, we MUST pay reserve (z must be 1).
# In optimization terms: If we choose NOT to pay reserve (z=0), the risk MUST be <= 0.2.
# If z=1, the risk constraint is relaxed (indicator is inactive).
risk_lhs = gp.quicksum(asset_data[i]['risk'] * x[i] for i in assets)
model.addGenConstrIndicator(z, 0, risk_lhs <= weighted_risk_factor_max * total_investment, name="Risk_Indicator_Constraint")

# 6. Solve the model
model.optimize()

# 7. Print results
if model.status == GRB.OPTIMAL:
    print("\nOptimal Solution Found:")
    for v in model.getVars():
        print(f"{v.VarName}: {v.X}")
    print(f"\nObjective Value: {model.objVal}")
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found.")