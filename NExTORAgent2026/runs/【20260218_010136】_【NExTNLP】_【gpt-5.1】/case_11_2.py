import gurobipy as gp
from gurobipy import GRB
import math

# =========================
# 1. Parameters (from Parameters List)
# =========================
tolerance = 0.05
G = 11500000.0
rho = 0.284
F_work = 300
tau_allowable = 80000
f_max = 4
delta_allowable = 0.5
C_range = [4, 12]
outer_diameter_max = 1.5
d_range = [0.1, 0.5]
D_range = [0.5, 6.0]
weight_constant = 1

pi = math.pi
Kf = (2.0 * pi * f_max) ** 2  # (2*pi*f_max)^2

# =========================
# 2. Create model
# =========================
model = gp.Model("Spring_Design_Optimization")
model.Params.NonConvex = 2

# =========================
# 3. Decision variables
# =========================
d = model.addVar(lb=d_range[0], ub=d_range[1], vtype=GRB.CONTINUOUS, name="d")
D = model.addVar(lb=D_range[0], ub=D_range[1], vtype=GRB.CONTINUOUS, name="D")
N = model.addVar(lb=1, vtype=GRB.INTEGER, name="N")  # total coils per context

# =========================
# 4. Auxiliary variables (all unrestricted as requested)
# =========================
C = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="C")              # spring index
Kw = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Kw")            # Wahl factor

num1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="num1")
den1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="den1")
inv_den1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="inv_den1")
term1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="term1")
invC = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="invC")
term2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="term2")

t = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="tau")            # shear stress
expr_tau = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="expr_tau")

k_var = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="k")          # stiffness
m_var = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="m")          # mass
r = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="r")              # k/m

delta = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="delta")      # max compression
W = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="W")              # weight proxy

# =========================
# 5. Constraints
# =========================

# 5.1 Spring index C relation and bounds: C = D/d, 4 <= C <= 12
model.addConstr(D == C * d, name="C_def")
model.addConstr(C >= C_range[0], name="C_min")
model.addConstr(C <= C_range[1], name="C_max")

# 5.2 Outer diameter: D + d <= 1.5
model.addConstr(D + d <= outer_diameter_max, name="outer_diameter")

# 5.3 Wahl factor Kw
model.addConstr(num1 == 4.0 * C - 1.0, name="num1_def")
model.addConstr(den1 == 4.0 * C - 4.0, name="den1_def")
model.addConstr(den1 * inv_den1 == 1.0, name="inv_den1_def")  # inv_den1 = 1/den1
model.addConstr(term1 == num1 * inv_den1, name="term1_def")
model.addConstr(C * invC == 1.0, name="invC_def")              # invC = 1/C
model.addConstr(term2 == 0.615 * invC, name="term2_def")
model.addConstr(Kw == term1 + term2, name="Kw_def")

# 5.4 Shear stress: tau = (8*F_work*D)/(pi*d^3) * Kw <= tau_allowable
model.addConstr(expr_tau == 8.0 * F_work * D * Kw, name="expr_tau_def")
model.addConstr(t * pi * d * d * d == expr_tau, name="tau_def")
model.addConstr(t <= tau_allowable, name="tau_limit")

# 5.5 Stiffness k: k = G d^4 / (8 D^3 N)
model.addConstr(G * d * d * d * d == 8.0 * D * D * D * N * k_var, name="k_def")

# 5.6 Mass m: m = rho * pi^2 * d^2 * D * (N+2) / 4
model.addConstr(
    m_var == rho * pi * pi * d * d * D * (N + 2) / 4.0,
    name="m_def"
)

# 5.7 Ratio r = k/m: k = r * m
model.addConstr(k_var == r * m_var, name="r_def")

# 5.8 Natural frequency: f = (1/(2*pi))*sqrt(k/m) <= f_max  <=>  r <= (2*pi*f_max)^2
model.addConstr(r <= Kf, name="freq_limit")

# 5.9 Compression: delta = 8 F_work D^3 N / (G d^4) >= delta_allowable
model.addConstr(
    8.0 * F_work * D * D * D * N == delta * G * d * d * d * d,
    name="delta_def"
)
model.addConstr(delta >= delta_allowable, name="delta_min")

# 5.10 Weight/volume proxy: W = (pi^2/4)*d^2*D*(N+2)
model.addConstr(
    W == (pi * pi / 4.0) * d * d * D * (N + 2),
    name="W_def"
)

# =========================
# 6. Objective
# =========================
model.setObjective(W * weight_constant, GRB.MINIMIZE)

# =========================
# 7. Optimize
# =========================
model.optimize()

# =========================
# 8. Output solution
# =========================
if model.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL] or (model.Status == GRB.INTERRUPTED and model.SolCount > 0):
    d_opt = d.X
    D_opt = D.X
    N_opt = N.X
    W_opt = W.X
    t_opt = t.X
    r_opt = r.X
    delta_opt = delta.X
    f_opt = (1.0 / (2.0 * pi)) * math.sqrt(max(r_opt, 0.0))

    print("Optimal design:")
    print(f"d (wire diameter)      = {d_opt:.6f} in")
    print(f"D (spring diameter)    = {D_opt:.6f} in")
    print(f"N (total coils)        = {N_opt:.6f}")
    print(f"W (weight proxy)       = {W_opt:.6f}")
    print(f"tau (shear stress)     = {t_opt:.6f} psi")
    print(f"f (natural frequency)  = {f_opt:.6f} Hz")
    print(f"delta (compression)    = {delta_opt:.6f} in")

    the_question_answer = W_opt
else:
    print("No feasible or no satisfactory solution found.")
    the_question_answer = float('nan')

print(f"FinalAnswer=【{the_question_answer}】")