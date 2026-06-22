import gurobipy as gp
import numpy as np

# 2. Define parameter matrices and data inputs
R_min = 0.11
mu = [0.12, 0.1, 0.15, 0.09]
Sigma = [[0.1, 0.02, 0.01, 0.005],
         [0.02, 0.05, 0.03, 0.01],
         [0.01, 0.03, 0.08, 0.02],
         [0.005, 0.01, 0.02, 0.03]]

# 3. Create decision variables
model = gp.Model("PortfolioVarianceMinimization")
w = model.addVars(4, lb=0, ub=1, name="w")

# 4. Create auxiliary substitution variables
# Variables for w_i^2
V = model.addVars(4, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="V")
# Variables for cross-terms w_i * w_j
P12 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="P12")
P13 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="P13")
P14 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="P14")
P23 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="P23")
P24 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="P24")
P34 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="P34")

# 5. Set up objective function
objective = (0.10 * V[0] + 0.05 * V[1] + 0.08 * V[2] + 0.03 * V[3] 
             + 0.04 * P12 + 0.02 * P13 + 0.01 * P14 
             + 0.06 * P23 + 0.02 * P24 + 0.04 * P34)
model.setObjective(objective, gp.GRB.MINIMIZE)

# 6. Add all constraints
# Handle non-convex terms
model.Params.NonConvex = 2

# Power constraints: V_i = w_i^2
for i in range(4):
    model.addGenConstrPow(w[i], V[i], 2, name=f"pow_constr_{i}")

# Bilinear constraints
model.addConstr(P12 == w[0] * w[1], name="bilinear_12")
model.addConstr(P13 == w[0] * w[2], name="bilinear_13")
model.addConstr(P14 == w[0] * w[3], name="bilinear_14")
model.addConstr(P23 == w[1] * w[2], name="bilinear_23")
model.addConstr(P24 == w[1] * w[3], name="bilinear_24")
model.addConstr(P34 == w[2] * w[3], name="bilinear_34")

# Expected return constraint
model.addConstr(gp.quicksum(mu[i] * w[i] for i in range(4)) >= R_min, name="return_constraint")

# Budget constraint (weights sum to 1)
model.addConstr(gp.quicksum(w[i] for i in range(4)) == 1, name="budget_constraint")

# 7. Solve the model and print results
model.optimize()

# Print optimal solution
if model.status == gp.GRB.OPTIMAL:
    print("Optimal portfolio weights:")
    for i in range(4):
        print(f"w{i+1}: {w[i].X:.4f}")
    
    # Calculate the portfolio variance (objective value)
    # Using the explicit formula for verification
    w_vals = np.array([w[i].X for i in range(4)])
    Sigma_np = np.array(Sigma)
    portfolio_variance = w_vals.T @ Sigma_np @ w_vals
    print(f"Portfolio variance (calculated): {portfolio_variance:.6f}")
    print(f"Objective value from Gurobi: {model.ObjVal:.6f}")
    
    # Final answer - the optimal portfolio variance
    # Use model.ObjVal as it should match the calculated variance
    print(f"FinalAnswer=【{model.ObjVal:.6f}】")
else:
    print(f"Optimization failed with status {model.status}")
    print(f"FinalAnswer=【Optimization failed】")