import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters (from Parameters List)
# =========================
alpha = 0.3        # Cobb-Douglas exponent for x
beta = 0.8         # Cobb-Douglas exponent for y
p_x = 4            # price of x
p_y = 2            # price of y
M = 100            # budget / big-M
N_threshold = 30   # quantity threshold for tip
tip_rate = 0.1     # tip rate

# =========================
# 2. Create model
# =========================
model = gp.Model("CobbDouglas_Consumer_Utility")

# Allow nonconvex general constraints (Pow, Mul)
model.Params.NonConvex = 2

# =========================
# 3. Decision variables (use given domains)
# =========================
# x: total units of commodity x purchased, integer, 0 ≤ x ≤ 25
x = model.addVar(vtype=GRB.INTEGER, lb=0, ub=25, name="x")

# y: total units of commodity y purchased, integer, 0 ≤ y ≤ 50
y = model.addVar(vtype=GRB.INTEGER, lb=0, ub=50, name="y")

# z: indicator of service-fee application (1 if x+y ≥ 31), binary
z = model.addVar(vtype=GRB.BINARY, name="z")

# u: auxiliary fee-base = z·(4x+2y), continuous, 0 ≤ u ≤ 100
u = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=100, name="u")

# =========================
# 4. Auxiliary substitution variables for nonlinear utility
#    (must be free: lb=-INF, ub=INF)
# =========================
# Xpow = x^alpha
Xpow = model.addVar(vtype=GRB.CONTINUOUS,
                    lb=-GRB.INFINITY, ub=GRB.INFINITY,
                    name="Xpow")

# Ypow = y^beta
Ypow = model.addVar(vtype=GRB.CONTINUOUS,
                    lb=-GRB.INFINITY, ub=GRB.INFINITY,
                    name="Ypow")

# U = Xpow * Ypow = x^alpha * y^beta
U = model.addVar(vtype=GRB.CONTINUOUS,
                 lb=-GRB.INFINITY, ub=GRB.INFINITY,
                 name="U")

# =========================
# 5. Nonlinear definitions (gen-constr)
# =========================
# Powers
model.addGenConstrPow(x, Xpow, alpha, name="Def_Xpow")
model.addGenConstrPow(y, Ypow, beta, name="Def_Ypow")

# Product for utility
model.addGenConstrMul(Xpow, Ypow, U, name="Def_U")

# =========================
# 6. Constraints (repaired mathematical model)
# =========================

# 6.1 Service-fee trigger as indicators:
# z = 1  -> x + y >= 31
model.addGenConstrIndicator(z, 1, x + y >= N_threshold + 1,
                            name="ServiceFee_Trigger_LB_Ind")
# z = 0  -> x + y <= 30
model.addGenConstrIndicator(z, 0, x + y <= N_threshold,
                            name="ServiceFee_Trigger_UB_Ind")

# 6.2 Definition of u = z·(4x + 2y) via indicators
# z = 1 -> u = 4x + 2y
model.addGenConstrIndicator(z, 1, u == p_x * x + p_y * y,
                            name="Aux_u_z1")
# z = 0 -> u = 0
model.addGenConstrIndicator(z, 0, u == 0,
                            name="Aux_u_z0")

# 6.3 Unified budget with possible service fee:
# 4x + 2y + tip_rate * u ≤ 100
model.addConstr(p_x * x + p_y * y + tip_rate * u <= M,
                name="Budget_WithFee")

# =========================
# 7. Objective function
# =========================
# Maximize Cobb-Douglas utility U = x^alpha * y^beta
model.setObjective(U, GRB.MAXIMIZE)

# =========================
# 8. Solve the model
# =========================
model.optimize()

# =========================
# 9. Print solution and final answer
# =========================
if model.status == GRB.OPTIMAL:
    x_opt = int(round(x.X))
    y_opt = int(round(y.X))
    z_opt = int(round(z.X))
    u_opt = u.X
    U_opt = U.X

    print(f"Optimal x (units of commodity x) = {x_opt}")
    print(f"Optimal y (units of commodity y) = {y_opt}")
    print(f"Service fee indicator z          = {z_opt}")
    print(f"Auxiliary u (= z·(4x+2y))        = {u_opt:.6f}")
    print(f"Maximum utility U                = {U_opt:.6f}")

    # The question's answer: optimal purchase plan and maximum utility
    final_answer = f"x={x_opt}, y={y_opt}, U_max={U_opt:.6f}"
else:
    final_answer = "No optimal solution found"

print(f"FinalAnswer=【{final_answer}】")