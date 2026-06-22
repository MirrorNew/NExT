import gurobipy as gp
from gurobipy import GRB

# ==========================
# 1. Define parameters (from Parameters List only)
# ==========================
T_opt = 80
C_opt = 50
Y_max = 100.0
a_T = 0.01
a_C = 0.02
T_min = 50
T_max = 100
C_min = 20
C_max = 80
T_parabolic_threshold = 100          # not explicitly used in model, but kept for completeness
difficulty_max = 100                 # not explicitly used in model, but kept for completeness
difficulty_mean = 75
difficulty_coefficient = 0.05
C_risk_threshold = 60                # not explicitly used in model, but kept for completeness
T_risk_threshold = 85                # not explicitly used in model, but kept for completeness
risk_coeff_T = 0.001
risk_coeff_C = 0.0431
w_difficulty = 0.1
w_risk = 0.5

# ==========================
# 2. Create model
# ==========================
model = gp.Model("Rhode_Island_Antibiotic_Optimization")

# Allow non-convex quadratic constraints due to general quadratic terms
model.Params.NonConvex = 2

# ==========================
# 3. Create decision variables
# ==========================
# Temperature and concentration
T = model.addVar(lb=T_min, ub=T_max, vtype=GRB.CONTINUOUS, name="T")
C = model.addVar(lb=C_min, ub=C_max, vtype=GRB.CONTINUOUS, name="C")

# Yield, difficulty, risk, and objective value
Y = model.addVar(lb=0.0, ub=Y_max, vtype=GRB.CONTINUOUS, name="Y")
D = model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="D")
H = model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="H")
Z = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Z")

# ==========================
# 4. Auxiliary substitution variables (unbounded as requested)
# ==========================
Tm80 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Tm80")  # (T - 80)
Tm75 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Tm75")  # (T - 75)
Tm50 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Tm50")  # (T - 50)
Cm50 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Cm50")  # (C - 50)

Q1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q1")      # (Tm80)^2
Q2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q2")      # (Cm50)^2
Q3 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q3")      # (Tm75)^2
Q4 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q4")      # (Tm50)^2

# ==========================
# 5. Objective function: maximize Z = Y - w_difficulty*D - w_risk*H
# ==========================
model.setObjective(Z, GRB.MAXIMIZE)

# ==========================
# 6. Constraints
# ==========================

# 6.1 Linear relationships for auxiliary variables
model.addConstr(Tm80 == T - T_opt,        name="def_Tm80")  # T - 80
model.addConstr(Tm75 == T - difficulty_mean, name="def_Tm75")  # T - 75
model.addConstr(Tm50 == T - T_min,        name="def_Tm50")  # T - 50
model.addConstr(Cm50 == C - C_opt,        name="def_Cm50")  # C - 50

# 6.2 Quadratic definitions via general constraints (powers)
model.addGenConstrPow(Tm80, Q1, 2.0, name="pow_Q1")   # Q1 = (Tm80)^2
model.addGenConstrPow(Cm50, Q2, 2.0, name="pow_Q2")   # Q2 = (Cm50)^2
model.addGenConstrPow(Tm75, Q3, 2.0, name="pow_Q3")   # Q3 = (Tm75)^2
model.addGenConstrPow(Tm50, Q4, 2.0, name="pow_Q4")   # Q4 = (Tm50)^2

# 6.3 Yield definition:
# Y = Y_max - a_T * Q1 - a_C * Q2
model.addConstr(
    Y == Y_max - a_T * Q1 - a_C * Q2,
    name="def_Y"
)

# 6.4 Difficulty definition:
# D = difficulty_coefficient * Q3
model.addConstr(
    D == difficulty_coefficient * Q3,
    name="def_D"
)

# 6.5 Risk factor definition:
# H = risk_coeff_T * Q4 + risk_coeff_C * C
model.addConstr(
    H == risk_coeff_T * Q4 + risk_coeff_C * C,
    name="def_H"
)

# 6.6 Objective decomposition:
# Z = Y - w_difficulty * D - w_risk * H
model.addConstr(
    Z == Y - w_difficulty * D - w_risk * H,
    name="def_Z"
)

# Temperature and concentration bounds are already set in variable definitions.
# Yield, D, H bounds are also in variable definitions, so no extra constraints needed.

# ==========================
# 7. Solve the model
# ==========================
model.optimize()

# ==========================
# 8. Print results
# ==========================
if model.status == GRB.OPTIMAL:
    T_opt_sol = T.X
    C_opt_sol = C.X
    Y_sol = Y.X
    D_sol = D.X
    H_sol = H.X
    Z_sol = Z.X

    print(f"Optimal Temperature T: {T_opt_sol:.6f} °C")
    print(f"Optimal Concentration C: {C_opt_sol:.6f} %")
    print(f"Yield Y: {Y_sol:.6f} kg/batch")
    print(f"Difficulty D: {D_sol:.6f}")
    print(f"Risk factor H: {H_sol:.6f}")
    print(f"Objective Z (performance): {Z_sol:.6f}")

    # According to the problem, the question is to find the best T and C settings
    # We will output them as the final answer in tuple form (T, C)
    print(f"FinalAnswer=【({T_opt_sol:.6f}, {C_opt_sol:.6f})】")
else:
    print("No optimal solution found.")
    print("FinalAnswer=【None】")