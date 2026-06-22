import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs
T = 15
num_RW = 2
num_S = 4

# Initial Stocks
IC_init = 1200
IR_init = [500, 400]  # RW1, RW2
IS_init = [350, 450, 500, 600]  # S1, S2, S3, S4

# Demands per period for stores
d = [50, 60, 70, 80]

# Safety Stock (30% of demand)
alpha = 0.3
SS = [alpha * dj for dj in d]

# Transportation Costs
cCR = [0.55, 0.22]  # CW to RW1, CW to RW2
# RW to Store (i, j)
cRS = [
    [0.22, 0.20, 0.32, 0.38],  # RW1 -> S1, S2, S3, S4
    [0.68, 0.52, 0.34, 0.10]   # RW2 -> S1, S2, S3, S4
]

# Capacities
CapCR = 1000
CapRS = 500

# Costs
ordering_cost = 30
h_CW_RW = 0.2
h_S = 0.6

# 2. Create the model
model = gp.Model("SupplyChainOptimization")

# 3. Create decision variables
y = model.addVars(num_RW, T, vtype=GRB.BINARY, name="y")  # Order decision at RW i in period t
z = model.addVars(num_RW, T, lb=0, ub=CapCR, name="z")  # CW to RW replenishment
w = model.addVars(num_RW, num_S, T, lb=0, ub=CapRS, name="w")  # RW to Store shipments

IC = model.addVars(T, lb=0, name="IC")  # CW inventory end-of-period
IR = model.addVars(num_RW, T, lb=0, name="IR")  # RW inventory end-of-period
IS = model.addVars(num_S, T, lb=0, name="IS")  # Store inventory end-of-period

# 4. Set up the objective function
obj_ordering = gp.quicksum(ordering_cost * y[i, t] for i in range(num_RW) for t in range(T))
obj_holding_CW = gp.quicksum(h_CW_RW * IC[t] for t in range(T))
obj_holding_RW = gp.quicksum(h_CW_RW * IR[i, t] for i in range(num_RW) for t in range(T))
obj_holding_S = gp.quicksum(h_S * IS[j, t] for j in range(num_S) for t in range(T))
obj_trans_CR = gp.quicksum(cCR[i] * z[i, t] for i in range(num_RW) for t in range(T))
obj_trans_RS = gp.quicksum(cRS[i][j] * w[i, j, t] for i in range(num_RW) for j in range(num_S) for t in range(T))

model.setObjective(obj_ordering + obj_holding_CW + obj_holding_RW + obj_holding_S + obj_trans_CR + obj_trans_RS, GRB.MINIMIZE)

# 5. Add constraints

# Inventory flow constraints
for t in range(T):
    # Central Warehouse flow
    if t == 0:
        model.addConstr(IC[t] == IC_init - gp.quicksum(z[i, t] for i in range(num_RW)))
    else:
        model.addConstr(IC[t] == IC[t-1] - gp.quicksum(z[i, t] for i in range(num_RW)))
    
    # Regional Warehouse flow
    for i in range(num_RW):
        if t == 0:
            model.addConstr(IR[i, t] == IR_init[i] + z[i, t] - gp.quicksum(w[i, j, t] for j in range(num_S)))
        else:
            model.addConstr(IR[i, t] == IR[i, t-1] + z[i, t] - gp.quicksum(w[i, j, t] for j in range(num_S)))
            
    # Retail Store flow
    for j in range(num_S):
        if t == 0:
            model.addConstr(IS[j, t] == IS_init[j] + gp.quicksum(w[i, j, t] for i in range(num_RW)) - d[j])
        else:
            model.addConstr(IS[j, t] == IS[j, t-1] + gp.quicksum(w[i, j, t] for i in range(num_RW)) - d[j])

# Indicator constraints for ordering and replenishment
for i in range(num_RW):
    for t in range(T):
        # Case 0: If y[i,t] = 0, then z[i,t] <= 0
        model.addGenConstrIndicator(y[i, t], 0, z[i, t] <= 0)
        # Case 1: If y[i,t] = 1, then z[i,t] >= 1e-6 (must have some replenishment)
        model.addGenConstrIndicator(y[i, t], 1, z[i, t] >= 1e-6)

# Safety Stock requirements
for j in range(num_S):
    for t in range(T):
        model.addConstr(IS[j, t] >= SS[j])

# Minimum order frequency for RW1 (index 0)
model.addConstr(gp.quicksum(y[0, t] for t in range(T)) >= 10)

# 6. Solve the model
model.optimize()

# 7. Print results
if model.status == GRB.OPTIMAL:
    print(f"Minimum Total Cost: {model.objVal}")
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found.")