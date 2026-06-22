import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and any other necessary packages.
# (Already imported above)

# 2. Define all parameter matrices and data inputs.
NumberOfPeriods = 12
AssetsPerPeriod = 6
Assets = ['X', 'Y', 'Z', 'M', 'N', 'O']
Stocks = ['X', 'Y', 'Z']
Bonds = ['M', 'N', 'O']

OverallRiskLimit = 6
MaxSingleStockProportion = 0.4
MaxSingleBondProportion = 0.6
MaxTotalStockProportion = 0.7
MinTotalBondProportion = 0.3
MinAssetsSelected = 5
AvgWeight_ZM_Limit = 0.4
MinExposure = 0.1

# Risk Index from Table 1
RiskIndex = {'X': 8, 'Y': 6, 'Z': 10, 'M': 2, 'N': 1, 'O': 3}

# Expected annual rate of return (%) from Table 2
# Structure: Index 0 is None (padding), Indices 1-12 correspond to periods.
# None indicates the asset is uninvestable in that period.
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

model = gp.Model("XinghuiInvestment")

# 3. Create decision variables.
w = {} # Proportion of funds
y = {} # Binary selection indicator

for t in range(1, NumberOfPeriods + 1):
    for asset in Assets:
        w[t, asset] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"w_{t}_{asset}")
        y[t, asset] = model.addVar(vtype=GRB.BINARY, name=f"y_{t}_{asset}")

# 4. Set up the objective function.
# Maximize total expected return multiple (decimal)
# Note: Input returns are in percent, so divide by 100 to get decimal.
obj_expr = gp.LinExpr()
for t in range(1, NumberOfPeriods + 1):
    for asset in Assets:
        # Check if asset is available
        val = CurrentExpectedReturn[t].get(asset)
        if val is not None:
            obj_expr += w[t, asset] * (val / 100.0)

model.setObjective(obj_expr, GRB.MAXIMIZE)

# 5. Add all constraints.

# Constraint 1: Full investment of funds
for t in range(1, NumberOfPeriods + 1):
    model.addConstr(gp.quicksum(w[t, asset] for asset in Assets) == 1.0, name=f"FullInvest_{t}")

# Constraint 2 & 3: Non-investable assets (based on Data Table 2)
# Specifically Z in Period 5 and 10 is None
for t in range(1, NumberOfPeriods + 1):
    for asset in Assets:
        if CurrentExpectedReturn[t].get(asset) is None:
            # Asset not available
            model.addConstr(w[t, asset] == 0, name=f"Unavail_w_{t}_{asset}")
            model.addConstr(y[t, asset] == 0, name=f"Unavail_y_{t}_{asset}")

# Constraint 4: Risk limit per period
for t in range(1, NumberOfPeriods + 1):
    risk_sum = gp.quicksum(w[t, asset] * RiskIndex[asset] for asset in Assets)
    model.addConstr(risk_sum <= OverallRiskLimit, name=f"RiskLimit_{t}")

# Constraint 5: Stock asset upper bound (<= 0.4)
for t in range(1, NumberOfPeriods + 1):
    for asset in Stocks:
        model.addConstr(w[t, asset] <= MaxSingleStockProportion, name=f"MaxStock_{t}_{asset}")

# Constraint 6: Bond asset upper bound (<= 0.6)
for t in range(1, NumberOfPeriods + 1):
    for asset in Bonds:
        model.addConstr(w[t, asset] <= MaxSingleBondProportion, name=f"MaxBond_{t}_{asset}")

# Constraint 7: Total stock proportion cap (<= 0.7)
for t in range(1, NumberOfPeriods + 1):
    model.addConstr(gp.quicksum(w[t, asset] for asset in Stocks) <= MaxTotalStockProportion, name=f"TotalStockCap_{t}")

# Constraint 8: Total bond proportion floor (>= 0.3)
for t in range(1, NumberOfPeriods + 1):
    model.addConstr(gp.quicksum(w[t, asset] for asset in Bonds) >= MinTotalBondProportion, name=f"TotalBondFloor_{t}")

# Constraint 9: Minimum number of assets per period (>= 5)
for t in range(1, NumberOfPeriods + 1):
    model.addConstr(gp.quicksum(y[t, asset] for asset in Assets) >= MinAssetsSelected, name=f"MinAssets_{t}")

# Constraint 10: Policy limit on Z and M
# Average weight of Z and M over 12 periods <= 40% (Total sum <= 0.4 * 12 = 4.8)
sum_ZM = gp.quicksum(w[t, 'Z'] + w[t, 'M'] for t in range(1, NumberOfPeriods + 1))
model.addConstr(sum_ZM <= AvgWeight_ZM_Limit * NumberOfPeriods, name="PolicyLimitZM")

# Constraint 11 & 12: Minimum exposure and Linking w and y
# Using Indicator Constraints as requested
for t in range(1, NumberOfPeriods + 1):
    for asset in Assets:
        # If y=1, then w >= 0.1
        model.addGenConstrIndicator(y[t, asset], 1, w[t, asset] >= MinExposure, name=f"Ind_MinExp_{t}_{asset}")
        # If y=0, then w <= 0 (effectively w=0 since w>=0)
        model.addGenConstrIndicator(y[t, asset], 0, w[t, asset] <= 0.0, name=f"Ind_Zero_{t}_{asset}")

# 6. Solve the model and print results.
model.optimize()

if model.Status == GRB.OPTIMAL:
    final_solution = model.ObjVal
    print(f"FinalAnswer=【{final_solution}】")
else:
    print("FinalAnswer=【No Solution】")