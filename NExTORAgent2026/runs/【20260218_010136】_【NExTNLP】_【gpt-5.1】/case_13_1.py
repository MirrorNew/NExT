import gurobipy as gp
from gurobipy import GRB

# ==========================
# 1. Define Parameters
# ==========================
R_min = 0.11
mu = [0.12, 0.10, 0.15, 0.09]
Sigma = [
    [0.10, 0.02, 0.01, 0.005],
    [0.02, 0.05, 0.03, 0.01],
    [0.01, 0.03, 0.08, 0.02],
    [0.005, 0.01, 0.02, 0.03]
]

n_assets = 4

# ==========================
# 2. Create Model
# ==========================
model = gp.Model("MeanVariance_MinVar")

# ==========================
# 3. Decision Variables
# ==========================
# Investment weights w_i in each asset, 0 <= w_i <= 1
w = model.addVars(n_assets, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="w")

# ==========================
# 4. Auxiliary Variables (not strictly needed here, but per required structure)
#    No nonlinear substitutions required for this quadratic model.
# ==========================
aux = model.addVars(n_assets, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="aux")
# Note: These auxiliary vars are not used in constraints or objective since
#       no substitutions are necessary for this convex quadratic problem.

# ==========================
# 5. Objective Function: minimize portfolio variance w^T Sigma w
# ==========================
quad_expr = gp.QuadExpr()
for i in range(n_assets):
    for j in range(n_assets):
        quad_expr += w[i] * Sigma[i][j] * w[j]

model.setObjective(quad_expr, GRB.MINIMIZE)

# ==========================
# 6. Constraints
# ==========================

# 6.1 Expected return constraint: sum(mu_i * w_i) >= R_min
model.addConstr(
    gp.quicksum(mu[i] * w[i] for i in range(n_assets)) >= R_min,
    name="ExpectedReturn"
)

# 6.2 Budget constraint: sum(w_i) = 1
model.addConstr(
    gp.quicksum(w[i] for i in range(n_assets)) == 1.0,
    name="Budget"
)

# 6.3 Non-negativity already handled via lb=0 in variable definition.
# No additional constraints required.

# ==========================
# 7. Solve Model
# ==========================
model.Params.OutputFlag = 0  # Turn off solver output for cleanliness
model.optimize()

# ==========================
# 8. Print Results
# ==========================
if model.status == GRB.OPTIMAL:
    print("Optimal solution found.")
    weights = [w[i].X for i in range(n_assets)]
    var_opt = model.ObjVal

    for i in range(n_assets):
        print(f"w_{i+1} = {weights[i]:.6f}")
    print(f"Optimal portfolio variance = {var_opt:.10f}")

    # Required final answer print
    print(f"FinalAnswer=【{var_opt}】")
else:
    print("No optimal solution found.")
    # In case of no optimal solution, still print something for FinalAnswer
    print("FinalAnswer=【NaN】")