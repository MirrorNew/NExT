import gurobipy as gp
from gurobipy import GRB

# Data from Parameters List
NumberOfPeriods = 12
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

# RiskIndex from Parameters List Table 1
RiskIndex = {'X': 8, 'Y': 6, 'Z': 10, 'M': 2, 'N': 1, 'O': 3}

# CurrentExpectedReturn (Table 2) - Given in percentages
# None represents assets that are uninvestable in specific periods
CurrentExpectedReturn = [
    None, # Index 0
    {'X': 11.5, 'Y': 9.8, 'Z': 14.2, 'M': 5.1, 'N': 3.9, 'O': 6.2},  # Period 1
    {'X': 12.3, 'Y': 10.2, 'Z': 15.1, 'M': 4.9, 'N': 4.1, 'O': 5.8}, # Period 2
    {'X': 11.8, 'Y': 9.5, 'Z': 14.7, 'M': 5.0, 'N': 4.0, 'O': 6.0},  # Period 3
    {'X': 12.0, 'Y': 10.0, 'Z': 15.3, 'M': 5.2, 'N': 3.8, 'O': 6.1}, # Period 4
    {'X': 12.1, 'Y': 10.1, 'Z': None, 'M': 5.0, 'N': 4.2, 'O': 5.9}, # Period 5
    {'X': 11.9, 'Y': 9.7, 'Z': 15.0, 'M': 5.3, 'N': 4.0, 'O': 6.3},  # Period 6
    {'X': 12.4, 'Y': 10.3, 'Z': 15.2, 'M': 5.1, 'N': 4.1, 'O': 6.0}, # Period 7
    {'X': 12.2, 'Y': 10.0, 'Z': 14.9, 'M': 5.0, 'N': 3.9, 'O': 6.2}, # Period 8
    {'X': 11.7, 'Y': 9.6, 'Z': 14.5, 'M': 4.8, 'N': 4.0, 'O': 6.1},  # Period 9
    {'X': 12.5, 'Y': 10.4, 'Z': None, 'M': 5.2, 'N': 4.3, 'O': 5.7}, # Period 10
    {'X': 12.0, 'Y': 10.1, 'Z': 15.4, 'M': 5.1, 'N': 4.2, 'O': 6.0}, # Period 11
    {'X': 11.6, 'Y': 9.9, 'Z': 14.8, 'M': 5.0, 'N': 4.1, 'O': 6.2}   # Period 12
]

# Create Gurobi Model
model = gp.Model("Xinghui_Investment_Optimization")

# Decision Variables
# w[t, i] represents the proportion of total funds allocated to asset i in period t
w = model.addVars(range(1, NumberOfPeriods + 1), Assets, lb=0.0, ub=1.0, name="w")
# y[t, i] is a binary indicator, 1 if asset i is selected in period t, 0 otherwise
y = model.addVars(range(1, NumberOfPeriods + 1), Assets, vtype=GRB.BINARY, name="y")

# Objective function: Maximize total annual returns (decimal form) across all 12 periods
model.setObjective(gp.quicksum(w[t, i] * (CurrentExpectedReturn[t][i] / 100.0) 
                               for t in range(1, NumberOfPeriods + 1) 
                               for i in Assets if CurrentExpectedReturn[t][i] is not None),
                   GRB.MAXIMIZE)

# Constraints
for t in range(1, NumberOfPeriods + 1):
    # Full investment of available funds (sum of asset weights must be 1.0)
    model.addConstr(gp.quicksum(w[t, i] for i in Assets) == 1.0)
    
    # Weighted risk index must be controlled within the upper limit (6.0)
    model.addConstr(gp.quicksum(w[t, i] * RiskIndex[i] for i in Assets) <= OverallRiskLimit)
    
    # Single stock asset proportion limits (not to exceed 40%)
    for i in Stocks:
        model.addConstr(w[t, i] <= MaxSingleStockProportion)
        
    # Single bond asset proportion limits (not to exceed 60%)
    for i in Bonds:
        model.addConstr(w[t, i] <= MaxSingleBondProportion)
        
    # Total stock asset proportion cap (must not exceed 70% in each period)
    model.addConstr(gp.quicksum(w[t, i] for i in Stocks) <= MaxTotalStockProportion)
    
    # Total bond asset proportion floor (must not be less than 30% in each period)
    model.addConstr(gp.quicksum(w[t, i] for i in Bonds) >= MinTotalBondProportion)
    
    # At least 5 assets must be invested in each period (MinAssetsSelected = 5)
    model.addConstr(gp.quicksum(y[t, i] for i in Assets) >= MinAssetsSelected)
    
    # Binary selection constraints and minimum exposure (at least 10% if selected)
    for i in Assets:
        if CurrentExpectedReturn[t][i] is None:
            # Asset cannot be invested in due to market or policy reasons
            model.addConstr(w[t, i] == 0)
            model.addConstr(y[t, i] == 0)
        else:
            # If asset is selected (y=1), its weight w must be at least MinExposure (0.1)
            model.addGenConstrIndicator(y[t, i], 1, w[t, i] >= MinExposure)
            # If asset is not selected (y=0), its weight w must be 0
            model.addGenConstrIndicator(y[t, i], 0, w[t, i] <= 0)
            # Link w and y: weight cannot exceed the binary selection flag
            model.addConstr(w[t, i] <= y[t, i])

# Policy limit constraint: The average weights of Stock Z and Bond M over 12 periods <= 40%
# Sum of weights over 12 periods <= 0.4 * 12 = 4.8
model.addConstr(gp.quicksum(w[t, 'Z'] for t in range(1, NumberOfPeriods + 1)) + 
                gp.quicksum(w[t, 'M'] for t in range(1, NumberOfPeriods + 1)) <= AvgWeight_ZM_Limit * NumberOfPeriods)

# Solve the optimization problem
model.optimize()

# Output the final optimized objective function value
if model.status == GRB.OPTIMAL:
    print(f"FinalAnswer=【{model.objVal}】")