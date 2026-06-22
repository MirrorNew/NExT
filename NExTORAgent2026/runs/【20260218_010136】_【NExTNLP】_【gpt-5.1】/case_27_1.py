import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Define all parameters
# =========================
peak_period_year = 2023
transmission_distance_threshold_km = 200
ieee_test_nodes = 14
annual_fuel_cost_saving_yuan = 120000000.0
emission_reduction_rate_percent = 0.3
n_power_sources = 2
n_load_nodes = 1
load_demand_MW = 100.0
loss_coefficient_P1 = 0.0005
loss_coefficient_P2 = 0.001
cost_coeff_P1_linear = 5.0
cost_coeff_P1_quadratic = 0.02
cost_coeff_P2_linear = 4.0
cost_coeff_P2_quadratic = 0.025

# =========================
# 2. Create model
# =========================
model = gp.Model("TwoSource_ELD_with_Nonlinear_Losses")

# Allow non-convex quadratic / general constraints
model.Params.NonConvex = 2

# =========================
# 3. Decision variables
# =========================
# Power outputs
P1 = model.addVar(lb=0.0, name="P1")  # Output of unit 1 (MW)
P2 = model.addVar(lb=0.0, name="P2")  # Output of unit 2 (MW)

# Line losses
L1 = model.addVar(lb=0.0, name="L1")  # Loss on line 1-3 (MW)
L2 = model.addVar(lb=0.0, name="L2")  # Loss on line 2-3 (MW)

# Generation costs
C1 = model.addVar(lb=0.0, name="C1")  # Cost of unit 1 ($/h)
C2 = model.addVar(lb=0.0, name="C2")  # Cost of unit 2 ($/h)

# =========================
# 4. Auxiliary substitution variables
# =========================
# Squares of power outputs
P1_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="P1_sq")
P2_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="P2_sq")

# Enforce P1_sq = P1^2 and P2_sq = P2^2 via general power constraints
model.addGenConstrPow(P1, P1_sq, 2.0, name="P1_square_def")
model.addGenConstrPow(P2, P2_sq, 2.0, name="P2_square_def")

# =========================
# 5. Objective function
# =========================
# Minimize total generation cost: Z = C1 + C2
model.setObjective(C1 + C2, GRB.MINIMIZE)

# =========================
# 6. Constraints
# =========================

# 6.1 Power balance: P1 + P2 - L1 - L2 = load_demand_MW
model.addConstr(P1 + P2 - L1 - L2 == load_demand_MW, name="Power_balance")

# 6.2 Loss definitions using auxiliary squares
# L1 = loss_coefficient_P1 * P1^2
model.addConstr(L1 == loss_coefficient_P1 * P1_sq, name="Loss1_definition")

# L2 = loss_coefficient_P2 * P2^2
model.addConstr(L2 == loss_coefficient_P2 * P2_sq, name="Loss2_definition")

# 6.3 Cost definitions using auxiliary squares
# C1 = 5*P1 + 0.02*P1^2
model.addConstr(
    C1 == cost_coeff_P1_linear * P1 + cost_coeff_P1_quadratic * P1_sq,
    name="Cost1_definition"
)

# C2 = 4*P2 + 0.025*P2^2
model.addConstr(
    C2 == cost_coeff_P2_linear * P2 + cost_coeff_P2_quadratic * P2_sq,
    name="Cost2_definition"
)

# (Nonnegativity already enforced via lb>=0 on P1, P2, L1, L2, C1, C2.)

# =========================
# 7. Solve model
# =========================
model.optimize()

# =========================
# 8. Print results
# =========================
if model.Status == GRB.OPTIMAL:
    print("Optimal solution found.")
    P1_opt = P1.X
    P2_opt = P2.X
    L1_opt = L1.X
    L2_opt = L2.X
    C1_opt = C1.X
    C2_opt = C2.X
    Z_opt = model.ObjVal

    print(f"P1 (MW) = {P1_opt:.6f}")
    print(f"P2 (MW) = {P2_opt:.6f}")
    print(f"L1 (MW) = {L1_opt:.6f}")
    print(f"L2 (MW) = {L2_opt:.6f}")
    print(f"C1 ($/h) = {C1_opt:.6f}")
    print(f"C2 ($/h) = {C2_opt:.6f}")
    print(f"Total cost Z ($/h) = {Z_opt:.6f}")

    # The question asks: determine the outputs P1, P2 of the two units.
    # So FinalAnswer will be reported as a tuple (P1, P2).
    print(f"FinalAnswer=【({P1_opt:.6f}, {P2_opt:.6f})】")
else:
    print(f"Optimization ended with status {model.Status}")
    # In a non-optimal case, still print a placeholder FinalAnswer
    print("FinalAnswer=【(None, None)】")