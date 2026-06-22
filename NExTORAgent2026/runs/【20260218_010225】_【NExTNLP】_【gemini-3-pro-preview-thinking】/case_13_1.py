import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs
R_min = 0.11
mu = [0.12, 0.10, 0.15, 0.09]
Sigma = [
    [0.10, 0.02, 0.01, 0.005],
    [0.02, 0.05, 0.03, 0.01],
    [0.01, 0.03, 0.08, 0.02],
    [0.005, 0.01, 0.02, 0.03]
]
num_assets = len(mu)

# 2. Create the model
model = gp.Model("Portfolio_Optimization")

# 3. Create decision variables
# w[i] represents the investment weight for asset i+1
w = model.addVars(num_assets, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="w")

# 4. Auxiliary substitution or indicator variables
# No auxiliary variables needed for standard QP objective formulation

# 5. Set up the objective function
# Minimize Portfolio Variance: w^T * Sigma * w
# We construct the quadratic expression directly
portfolio_variance = gp.QuadExpr()
for i in range(num_assets):
    for j in range(num_assets):
        portfolio_variance += w[i] * Sigma[i][j] * w[j]

model.setObjective(portfolio_variance, GRB.MINIMIZE)

# 6. Add constraints
# Constraint 1: Expected return constraint (sum(mu_i * w_i) >= R_min)
model.addConstr(gp.quicksum(mu[i] * w[i] for i in range(num_assets)) >= R_min, name="Min_Expected_Return")

# Constraint 2: Budget constraint (sum(w_i) = 1)
model.addConstr(gp.quicksum(w[i] for i in range(num_assets)) == 1.0, name="Budget")

# Constraint 3: Non-negativity is handled by the lb=0.0 in addVars

# 7. Solve the model and print results
model.optimize()

if model.Status == GRB.OPTIMAL:
    print(f"\nOptimal Portfolio Variance: {model.ObjVal}")
    print("Optimal Weights:")
    for i in range(num_assets):
        print(f"  Asset {i+1}: {w[i].X:.4f}")
    
    # Output the final answer in the requested format
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimization was stopped with status " + str(model.Status))