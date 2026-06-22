import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Define parameters
# =========================
alpha = 0.5       # α
beta = 0.7        # β
mu = 1            # μ
A_t = 20          # A_t
total_budget = 1000
w = 50            # labor cost
r = 100           # capital cost

# =========================
# 2. Create model
# =========================
model = gp.Model("CobbDouglas_Production")

# Allow non-convex quadratic/bilinear + power constraints
model.Params.NonConvex = 2

# =========================
# 3. Decision variables
# =========================
L = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="L")  # Labor
K = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="K")  # Capital

# =========================
# 4. Auxiliary variables
# =========================
# Q: output
Q = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="Q")

# L_root ~ L^0.5
L_root = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="L_root")

# K_root ~ K^0.7
K_root = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="K_root")

# =========================
# 5. Objective: maximize Q
# =========================
model.setObjective(Q, GRB.MAXIMIZE)

# =========================
# 6. Constraints
# =========================

# Budget constraint: 50 L + 100 K <= 1000
model.addConstr(w * L + r * K <= total_budget, name="Budget")

# Non-negativity (explicit)
model.addConstr(L >= 0, name="L_nonneg")
model.addConstr(K >= 0, name="K_nonneg")
model.addConstr(Q >= 0, name="Q_nonneg")

# Power constraints: L_root = L^0.5, K_root = K^0.7
model.addGenConstrPow(L, L_root, alpha, name="L_pow")
model.addGenConstrPow(K, K_root, beta, name="K_pow")

# Production function: Q = A_t * L^0.5 * K^0.7
# Implemented as: Q == A_t * L_root * K_root
model.addConstr(Q == A_t * L_root * K_root, name="Production")

# =========================
# 7. Solve model
# =========================
model.optimize()

# =========================
# 8. Print results
# =========================
if model.status == GRB.OPTIMAL:
    L_opt = L.X
    K_opt = K.X
    Q_opt = Q.X

    print(f"Optimal labor L* = {L_opt:.6f}")
    print(f"Optimal capital K* = {K_opt:.6f}")
    print(f"Maximum output Q* = {Q_opt:.6f}")
else:
    print("No optimal solution found.")
    L_opt = float('nan')
    K_opt = float('nan')
    Q_opt = float('nan')

# According to the question, the requested final answer is the calculated output Q(L,K).
print(f"FinalAnswer=【{Q_opt}】")