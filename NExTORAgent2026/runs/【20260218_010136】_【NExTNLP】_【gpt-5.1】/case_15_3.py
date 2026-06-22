import gurobipy as gp
from gurobipy import GRB

# 1. Parameters (from Parameters List)
alpha = 0.3
beta = 0.8
p_x = 4
p_y = 2
M = 100
N_threshold = 30
tip_rate = 0.1

# 2. Create model
model = gp.Model("CobbDouglas_Consumer_Utility")

# Allow nonconvex general constraints
model.Params.NonConvex = 2

# 3. Decision variables (from Context)
x = model.addVar(vtype=GRB.INTEGER, lb=0, ub=25, name="x")   # 0 ≤ x ≤ 25
y = model.addVar(vtype=GRB.INTEGER, lb=0, ub=50, name="y")   # 0 ≤ y ≤ 50
z = model.addVar(vtype=GRB.BINARY, name="z")                 # {0,1}
u = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=100, name="u")  # 0 ≤ u ≤ 100

# 4. Auxiliary substitution variables for nonlinear utility (free)
Xpow = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Xpow")
Ypow = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Ypow")
U = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="U")

# 5. Nonlinear definitions
model.addGenConstrPow(x, Xpow, alpha, name="Def_Xpow")
model.addGenConstrPow(y, Ypow, beta, name="Def_Ypow")
model.addGenConstrMul(Xpow, Ypow, U, name="Def_U")

# 6. Constraints (validated math model)

# Budget_NoFee: 4 x + 2 y ≤ 100
model.addConstr(p_x * x + p_y * y <= M, name="Budget_NoFee")

# ServiceFee_Trigger_UB: x + y ≤ 30 + 20 z
model.addConstr(x + y <= N_threshold + 20 * z, name="ServiceFee_Trigger_UB")

# ServiceFee_Trigger_LB: x + y ≥ 31 z
model.addConstr(x + y >= (N_threshold + 1) * z, name="ServiceFee_Trigger_LB")

# Aux_u_Upper_xy: u ≤ 4 x + 2 y
model.addConstr(u <= p_x * x + p_y * y, name="Aux_u_Upper_xy")

# Aux_u_Upper_Mz: u ≤ 100 z
model.addConstr(u <= M * z, name="Aux_u_Upper_Mz")

# Aux_u_Lower: u ≥ 4 x + 2 y – 100*(1–z)
model.addConstr(u >= p_x * x + p_y * y - M * (1 - z), name="Aux_u_Lower")

# Budget_WithFee: 4 x + 2 y + 0.1 u ≤ 100
model.addConstr(p_x * x + p_y * y + tip_rate * u <= M, name="Budget_WithFee")

# 7. Objective
model.setObjective(U, GRB.MAXIMIZE)

# 8. Optimize
model.optimize()

# 9. Output results
if model.status == GRB.OPTIMAL:
    x_opt = int(round(x.X))
    y_opt = int(round(y.X))
    z_opt = int(round(z.X))
    u_opt = u.X
    U_opt = U.X

    print(f"x = {x_opt}")
    print(f"y = {y_opt}")
    print(f"z = {z_opt}")
    print(f"u = {u_opt}")
    print(f"U = {U_opt}")

    final_answer = f"x={x_opt}, y={y_opt}, U_max={U_opt:.6f}"
else:
    final_answer = "No optimal solution found"

print(f"FinalAnswer=【{final_answer}】")