import gurobipy as gp
from gurobipy import GRB

# ============================
# 1. Define parameters (from Parameters List)
# ============================
investment_limit = 10.0
F_linear_coeff = 5.0
F_quad_coeff = -0.01
S_linear_coeff = 10.0
startup_consumption = 5.0
P_threshold = 2.0
consumption_quad_coeff = 1.0

# ============================
# 2. Create model
# ============================
model = gp.Model("Kazdel_Renewable_Investment_Optimization")

# Allow non-convex quadratic constraints (needed for Pow and indicator-combos)
model.Params.NonConvex = 2

# ============================
# 3. Decision variables
# ============================
# Investments
W = model.addVar(lb=0.0, ub=investment_limit, vtype=GRB.CONTINUOUS, name="W")  # wind investment
P = model.addVar(lb=0.0, ub=investment_limit, vtype=GRB.CONTINUOUS, name="P")  # PV investment

# Generation-related variables
F = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="F")  # wind generation
S = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="S")  # PV generation before deductions
C = model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="C")            # PV infrastructure consumption

# ============================
# 4. Auxiliary substitution and indicator variables
# ============================
# Wind quadratic term: Q_W2 = W^2
Q_W2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q_W2")

# For max{P - P_threshold, 0}: u = max{P - 2, 0}
u = model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="u")

# Quadratic of u: z = u^2
z = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z")

# Binary indicator: y = 0 if P <= 2, y = 1 if P >= 2
y = model.addVar(vtype=GRB.BINARY, name="y")

# ============================
# 5. Objective function
# Z = F + S - startup_consumption - C
# ============================
Z = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Z")

model.addConstr(Z == F + S - startup_consumption - C, name="Z_definition")

model.setObjective(Z, GRB.MAXIMIZE)

# ============================
# 6. Constraints
# ============================

# 6.1 Investment limit
model.addConstr(W + P <= investment_limit, name="budget")

# 6.2 Wind power generation: F = 5W - 0.01 W^2, using auxiliary Q_W2
# Q_W2 = W^2
model.addGenConstrPow(W, Q_W2, 2.0, name="wind_square")
# F = F_linear_coeff * W + F_quad_coeff * Q_W2
model.addConstr(
    F == F_linear_coeff * W + F_quad_coeff * Q_W2,
    name="wind_gen"
)

# 6.3 PV generation: S = 10P
model.addConstr(
    S == S_linear_coeff * P,
    name="pv_gen"
)

# 6.4 Max structure u = max{P - P_threshold, 0} via indicator constraints

# Case y = 0: P <= P_threshold, so u = 0
model.addGenConstrIndicator(
    y, 0, u == 0.0, name="u_zero_if_P_le_threshold"
)
model.addGenConstrIndicator(
    y, 0, P <= P_threshold, name="P_le_threshold_if_y0"
)

# Case y = 1: P >= P_threshold, so u = P - P_threshold
model.addGenConstrIndicator(
    y, 1, u == P - P_threshold, name="u_eq_P_minus_threshold_if_P_ge_threshold"
)
model.addGenConstrIndicator(
    y, 1, P >= P_threshold, name="P_ge_threshold_if_y1"
)

# 6.5 Quadratic relation for infrastructure consumption:
# z = u^2, C = consumption_quad_coeff * z
model.addGenConstrPow(u, z, 2.0, name="u_square")
model.addConstr(
    C == consumption_quad_coeff * z,
    name="C_eq_z"
)

# ============================
# 7. Solve the model and print results
# ============================
model.optimize()

if model.status == GRB.OPTIMAL:
    W_val = W.X
    P_val = P.X
    F_val = F.X
    S_val = S.X
    C_val = C.X
    Z_val = Z.X

    print(f"Optimal wind investment W (billion yuan): {W_val:.6f}")
    print(f"Optimal PV investment P (billion yuan): {P_val:.6f}")
    print(f"Wind generation F (billion kWh): {F_val:.6f}")
    print(f"PV generation before deductions S (billion kWh): {S_val:.6f}")
    print(f"PV infrastructure consumption C (billion kWh): {C_val:.6f}")
    print(f"Startup consumption (billion kWh): {startup_consumption:.6f}")
    print(f"Total net generation Z (billion kWh): {Z_val:.6f}")

    # FinalAnswer is the total maximum net power generation
    print(f"FinalAnswer=【{Z_val:.6f}】")
else:
    # If not optimal, print status and set FinalAnswer as NaN
    print(f"Optimization ended with status {model.status}")
    print("FinalAnswer=【nan】")