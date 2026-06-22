import gurobipy as gp
from gurobipy import GRB
import math

# =========================
# 1. Parameters (from Parameters List)
# =========================
tolerance = 0.05
G = 11500000.0          # shear modulus (psi)
rho = 0.284             # density (lb/in^3) – used in mass, but weight_constant=1 so objective uses volume form
F_work = 300            # working load (lbf)
tau_allowable = 80000   # allowable shear stress (psi)
f_max = 4               # maximum natural frequency (Hz)
delta_allowable = 0.5   # minimum allowable compression (in)
C_range = [4, 12]       # spring index range
outer_diameter_max = 1.5
d_range = [0.1, 0.5]
D_range = [0.5, 6.0]
weight_constant = 1

pi = math.pi

# Precompute constant used in natural-frequency constraint: (2*pi*f_max)^2
Kf = (2.0 * pi * f_max) ** 2

# =========================
# 2. Create model
# =========================
model = gp.Model("Spring_Design_Optimization")
model.Params.NonConvex = 2  # allow general nonlinear constraints

# =========================
# 3. Decision variables
# =========================
# d: wire diameter
d = model.addVar(lb=d_range[0], ub=d_range[1], vtype=GRB.CONTINUOUS, name="d")

# D: spring center (mean) diameter
D = model.addVar(lb=D_range[0], ub=D_range[1], vtype=GRB.CONTINUOUS, name="D")

# N: total number of coils (effective + end supports), integer, N >= 1
N = model.addVar(lb=1, vtype=GRB.INTEGER, name="N")

# =========================
# 4. Auxiliary / substitution variables
# =========================
# C: spring index = D/d (we enforce via D == C * d)
C = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="C")

# Wahl factor Kw and its components
Kw = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Kw")
num1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="num1")
den1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="den1")
inv_den1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="inv_den1")
term1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="term1")
invC = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="invC")
term2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="term2")

# Shear stress tau and its numerator expression
t = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="tau")
expr_tau = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="expr_tau")

# Spring stiffness k
k_var = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="k")

# Spring mass m (for frequency constraint)
m_var = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="m")

# Ratio r = k/m
r = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="r")

# Maximum compression delta
delta = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="delta")

# Weight/volume proxy W
W = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="W")

# =========================
# 5. Constraints
# =========================

# 5.1 Spring index relation: D = C * d (so C = D/d without division)
model.addConstr(D == C * d, name="C_def")

# 5.2 Winding ratio bounds: 4 <= C <= 12
model.addConstr(C >= C_range[0], name="C_min")
model.addConstr(C <= C_range[1], name="C_max")

# 5.3 Outer diameter constraint: D + d <= outer_diameter_max
model.addConstr(D + d <= outer_diameter_max, name="outer_diameter")

# 5.4 Wahl factor Kw
# num1 = 4*C - 1
model.addConstr(num1 == 4.0 * C - 1.0, name="num1_def")
# den1 = 4*C - 4
model.addConstr(den1 == 4.0 * C - 4.0, name="den1_def")
# den1 * inv_den1 = 1  => inv_den1 = 1 / den1
model.addConstr(den1 * inv_den1 == 1.0, name="inv_den1_def")
# term1 = num1 * inv_den1 = (4C-1)/(4C-4)
model.addConstr(term1 == num1 * inv_den1, name="term1_def")
# C * invC = 1 => invC = 1 / C
model.addConstr(C * invC == 1.0, name="invC_def")
# term2 = 0.615 * invC = 0.615 / C
model.addConstr(term2 == 0.615 * invC, name="term2_def")
# Kw = term1 + term2
model.addConstr(Kw == term1 + term2, name="Kw_def")

# 5.5 Shear stress:
# expr_tau = 8 * F_work * D * Kw
model.addConstr(expr_tau == 8.0 * F_work * D * Kw, name="expr_tau_def")
# t * pi * d^3 = expr_tau  -> t = (8*F_work*D*Kw)/(pi*d^3)
model.addConstr(t * pi * d * d * d == expr_tau, name="tau_def")
# Shear stress constraint: t <= tau_allowable
model.addConstr(t <= tau_allowable, name="tau_limit")

# 5.6 Spring stiffness k: G * d^4 = 8 * D^3 * N * k
# (matches k = G d^4 / (8 D^3 N))
model.addConstr(G * d * d * d * d == 8.0 * D * D * D * N * k_var, name="k_def")

# 5.7 Spring mass m:
# m = rho * pi^2 * d^2 * D * (N+2) / 4
# note: no variable in denominator, so we can code it directly
model.addConstr(
    m_var == rho * pi * pi * d * d * D * (N + 2) / 4.0,
    name="m_def"
)

# 5.8 Ratio r = k/m: enforce k = r * m  -> r = k/m
model.addConstr(k_var == r * m_var, name="r_def")

# 5.9 Natural frequency constraint:
# f = (1/(2*pi))*sqrt(k/m) <= f_max
# Equivalent: k/m <= (2*pi*f_max)^2  -> r <= Kf
model.addConstr(r <= Kf, name="freq_limit")

# 5.10 Maximum compression delta:
# delta = 8 * F_work * D^3 * N / (G * d^4)
# Implement without division: 8 F_work D^3 N = delta * G d^4
model.addConstr(
    8.0 * F_work * D * D * D * N == delta * G * d * d * d * d,
    name="delta_def"
)
# Minimum compression constraint: delta >= delta_allowable
model.addConstr(delta >= delta_allowable, name="delta_min")

# =========================
# 6. Objective: minimize weight
# W = (pi^2/4)*d^2*D*(N+2)
# =========================
model.addConstr(
    W == (pi * pi / 4.0) * d * d * D * (N + 2),
    name="W_def"
)

model.setObjective(W * weight_constant, GRB.MINIMIZE)

# =========================
# 7. Optimize
# =========================
model.optimize()

# =========================
# 8. Print results
# =========================
if (model.status == GRB.OPTIMAL) or (model.status == GRB.INTERRUPTED and model.SolCount > 0):
    d_opt = d.X
    D_opt = D.X
    N_opt = N.X
    W_opt = W.X
    t_opt = t.X
    r_opt = r.X
    delta_opt = delta.X

    # Natural frequency from r: f = (1/(2*pi))*sqrt(r)
    f_opt = (1.0 / (2.0 * pi)) * math.sqrt(max(r_opt, 0.0))

    print("Optimal design found:")
    print(f"  d (wire diameter)      = {d_opt:.6f} in")
    print(f"  D (spring diameter)    = {D_opt:.6f} in")
    print(f"  N (total coils)        = {N_opt:.6f}")
    print(f"  Weight proxy W         = {W_opt:.66f}")
    print(f"  Shear stress tau       = {t_opt:.6f} psi")
    print(f"  Natural frequency f    = {f_opt:.6f} Hz")
    print(f"  Max compression delta  = {delta_opt:.6f} in")

    # The main question is to minimize the weight of the spring -> answer is optimal W
    the_question_answer = W_opt
else:
    print("No feasible solution found or optimization was not successful.")
    the_question_answer = float('nan')

# Final required output
print(f"FinalAnswer=【{the_question_answer}】")