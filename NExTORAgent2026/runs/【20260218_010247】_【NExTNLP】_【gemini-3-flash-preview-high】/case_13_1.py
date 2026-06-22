import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs
R_min = 0.11
mu = [0.12, 0.1, 0.15, 0.09]
Sigma = [
    [0.1, 0.02, 0.01, 0.005],
    [0.02, 0.05, 0.03, 0.01],
    [0.01, 0.03, 0.08, 0.02],
    [0.005, 0.01, 0.02, 0.03]
]

# Create model
model = gp.Model("PortfolioOptimization")

# 2. Create decision variables
# Investment weights w1, w2, w3, w4
w1 = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="w1")
w2 = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="w2")
w3 = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="w3")
w4 = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="w4")

# 3. Create auxiliary substitution variables
# Squared variables
w1_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w1_sq")
w2_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w2_sq")
w3_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w3_sq")
w4_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w4_sq")

# Interaction product variables
w1w2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w1w2")
w1w3 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w1w3")
w1w4 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w1w4")
w2w3 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w2w3")
w2w4 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w2w4")
w3w4 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w3w4")

# 4. Add auxiliary constraints for nonlinear terms
model.Params.NonConvex = 2
model.addGenConstrPow(w1, w1_sq, 2)
model.addGenConstrPow(w2, w2_sq, 2)
model.addGenConstrPow(w3, w3_sq, 2)
model.addGenConstrPow(w4, w4_sq, 2)

model.addConstr(w1w2 == w1 * w2)
model.addConstr(w1w3 == w1 * w3)
model.addConstr(w1w4 == w1 * w4)
model.addConstr(w2w3 == w2 * w3)
model.addConstr(w2w4 == w2 * w4)
model.addConstr(w3w4 == w3 * w4)

# 5. Set up the objective function
# Objective: minimize wᵀ Σ w
# Explicitly: Σ11*w1^2 + Σ22*w2^2 + Σ33*w3^2 + Σ44*w4^2 
#           + (Σ12+Σ21)*w1w2 + (Σ13+Σ31)*w1w3 + (Σ14+Σ41)*w1w4
#           + (Σ23+Σ32)*w2w3 + (Σ24+Σ42)*w2w4 + (Σ34+Σ43)*w3w4
objective = (
    Sigma[0][0] * w1_sq + Sigma[1][1] * w2_sq + Sigma[2][2] * w3_sq + Sigma[3][3] * w4_sq +
    (Sigma[0][1] + Sigma[1][0]) * w1w2 + 
    (Sigma[0][2] + Sigma[2][0]) * w1w3 + 
    (Sigma[0][3] + Sigma[3][0]) * w1w4 +
    (Sigma[1][2] + Sigma[2][1]) * w2w3 + 
    (Sigma[1][3] + Sigma[3][1]) * w2w4 + 
    (Sigma[2][3] + Sigma[3][2]) * w3w4
)
model.setObjective(objective, GRB.MINIMIZE)

# 6. Add all constraints
# Expected return constraint: Σ mu_i * w_i >= R_min
model.addConstr(mu[0] * w1 + mu[1] * w2 + mu[2] * w3 + mu[3] * w4 >= R_min, "ReturnConstraint")

# Budget constraint: Σ w_i = 1
model.addConstr(w1 + w2 + w3 + w4 == 1, "BudgetConstraint")

# 7. Solve the model and print results
model.optimize()

if model.status == GRB.OPTIMAL:
    optimal_variance = model.objVal
    print(f"Optimal Portfolio Variance: {optimal_variance}")
    print(f"FinalAnswer=【{optimal_variance}】")