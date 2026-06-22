import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Define parameters
# =========================
saturation_threshold_TV = 600000
expansion_factor_weak_channel = 3
cost_reduction_percentage = 18.7
total_budget = 100
max_investment = {'A': 60, 'B': 50, 'C': 50}
revenue_coefficients = {'A': 30.23, 'B': 24.36, 'C': 20.12}
extra_investment_factor = 3
max_total_budget = 100

# =========================
# 2. Create model
# =========================
model = gp.Model("RhodeIsland_MMM_Optimization")

# Allow nonconvex constructs (Pow + binaries)
model.Params.NonConvex = 2

# =========================
# 3. Decision variables
# =========================
# Primary investments
x_A = model.addVar(lb=0.0, ub=max_investment['A'], vtype=GRB.CONTINUOUS, name="x_A")
x_B = model.addVar(lb=0.0, ub=max_investment['B'], vtype=GRB.CONTINUOUS, name="x_B")
x_C = model.addVar(lb=0.0, ub=max_investment['C'], vtype=GRB.CONTINUOUS, name="x_C")

# Secondary (effect-amplifier) investments
E_A = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="E_A")
E_B = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="E_B")
E_C = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="E_C")

# Binary indicators: 1 if the channel is the unique lowest primary investment
delta_A = model.addVar(vtype=GRB.BINARY, name="delta_A")
delta_B = model.addVar(vtype=GRB.BINARY, name="delta_B")
delta_C = model.addVar(vtype=GRB.BINARY, name="delta_C")

# Revenues
y_A = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="y_A")
y_B = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="y_B")
y_C = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="y_C")

# =========================
# 4. Auxiliary substitution variables
#    (domain set to -INF..INF as required)
# =========================
# Total spend per channel s_i = x_i + E_i
s_A = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="s_A")
s_B = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="s_B")
s_C = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="s_C")

# Square-root arguments: r_i = sqrt(s_i)
r_A = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="r_A")
r_B = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="r_B")
r_C = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="r_C")

# =========================
# 5. Objective: maximize total revenue
# =========================
model.setObjective(y_A + y_B + y_C, GRB.MAXIMIZE)

# =========================
# 6. Constraints
# =========================

# C1 & C3: TV saturation threshold and max investment
model.addConstr(x_A <= saturation_threshold_TV, name="TV_saturation_threshold")
model.addConstr(x_A <= max_investment['A'], name="max_invest_A")

# C4 & C5: Max investments for B and C
model.addConstr(x_B <= max_investment['B'], name="max_invest_B")
model.addConstr(x_C <= max_investment['C'], name="max_invest_C")

# C2: Initial budget limit (primary investments)
model.addConstr(x_A + x_B + x_C <= total_budget, name="initial_budget_limit")

# C6: Exactly one lowest-investment channel
model.addConstr(delta_A + delta_B + delta_C == 1, name="one_lowest_channel")

# C7–C9: Minimum-channel identification using indicator constraints
# If delta_A = 1 then x_A <= x_B and x_A <= x_C
model.addGenConstrIndicator(delta_A, 1, x_A <= x_B, name="A_min_leq_B")
model.addGenConstrIndicator(delta_A, 1, x_A <= x_C, name="A_min_leq_C")

# If delta_B = 1 then x_B <= x_A and x_B <= x_C
model.addGenConstrIndicator(delta_B, 1, x_B <= x_A, name="B_min_leq_A")
model.addGenConstrIndicator(delta_B, 1, x_B <= x_C, name="B_min_leq_C")

# If delta_C = 1 then x_C <= x_A and x_C <= x_B
model.addGenConstrIndicator(delta_C, 1, x_C <= x_A, name="C_min_leq_A")
model.addGenConstrIndicator(delta_C, 1, x_C <= x_B, name="C_min_leq_B")

# C10: Secondary investment definition via indicator constraints
# Channel A
model.addGenConstrIndicator(delta_A, 1, E_A == extra_investment_factor * x_A,
                            name="E_A_when_A_is_min")
model.addGenConstrIndicator(delta_A, 0, E_A == 0,
                            name="E_A_when_A_not_min")

# Channel B
model.addGenConstrIndicator(delta_B, 1, E_B == extra_investment_factor * x_B,
                            name="E_B_when_B_is_min")
model.addGenConstrIndicator(delta_B, 0, E_B == 0,
                            name="E_B_when_B_not_min")

# Channel C
model.addGenConstrIndicator(delta_C, 1, E_C == extra_investment_factor * x_C,
                            name="E_C_when_C_is_min")
model.addGenConstrIndicator(delta_C, 0, E_C == 0,
                            name="E_C_when_C_not_min")

# C11: Final budget including secondary investments
model.addConstr(
    x_A + x_B + x_C + E_A + E_B + E_C <= max_total_budget,
    name="final_total_budget"
)

# C12: Revenue function definitions using auxiliary variables
# s_i = x_i + E_i
model.addConstr(s_A == x_A + E_A, name="s_A_def")
model.addConstr(s_B == x_B + E_B, name="s_B_def")
model.addConstr(s_C == x_C + E_C, name="s_C_def")

# r_i = sqrt(s_i) via power constraints
# Note: syntax: addGenConstrPow(x, y, a) enforces y = x^a
model.addGenConstrPow(s_A, r_A, 0.5, name="sqrt_A")
model.addGenConstrPow(s_B, r_B, 0.5, name="sqrt_B")
model.addGenConstrPow(s_C, r_C, 0.5, name="sqrt_C")

# y_i = coeff_i * r_i
model.addConstr(y_A == revenue_coefficients['A'] * r_A, name="y_A_def")
model.addConstr(y_B == revenue_coefficients['B'] * r_B, name="y_B_def")
model.addConstr(y_C == revenue_coefficients['C'] * r_C, name="y_C_def")

# C13: Nonnegativity implicitly handled by variable bounds for primary/secondary/revenues
# (x_A, x_B, x_C, E_A, E_B, E_C, y_A, y_B, y_C ≥ 0)
# delta_A, delta_B, delta_C ∈ {0,1} already via vtype=BINARY

# =========================
# 7. Solve the model
# =========================
model.optimize()

# =========================
# 8. Print results
# =========================
if model.SolCount > 0:
    xA_val = x_A.X
    xB_val = x_B.X
    xC_val = x_C.X
    EA_val = E_A.X
    EB_val = E_B.X
    EC_val = E_C.X
    yA_val = y_A.X
    yB_val = y_B.X
    yC_val = y_C.X
    Z_val = yA_val + yB_val + yC_val

    print("Optimal solution found:")
    print(f"x_A (primary TV)      = {xA_val:.6f}")
    print(f"x_B (primary social)  = {xB_val:.6f}")
    print(f"x_C (primary radio)   = {xC_val:.6f}")
    print(f"E_A (secondary TV)    = {EA_val:.6f}")
    print(f"E_B (secondary social)= {EB_val:.6f}")
    print(f"E_C (secondary radio) = {EC_val:.6f}")
    print(f"y_A (revenue A)       = {yA_val:.6f}")
    print(f"y_B (revenue B)       = {yB_val:.6f}")
    print(f"y_C (revenue C)       = {yC_val:.6f}")
    print(f"Total revenue Z       = {Z_val:.6f}")
else:
    Z_val = float('nan')
    print("No feasible solution found.")

# =========================
# 9. FinalAnswer output
#     The question asks: "Give the maximum revenue after calculation."
#     So FinalAnswer is the optimal objective value Z.
# =========================
print(f"FinalAnswer=【{Z_val}】")