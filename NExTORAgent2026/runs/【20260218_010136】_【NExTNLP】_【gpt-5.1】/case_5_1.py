import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Define all parameters
# =========================
assets = ['A', 'B']                 # from Parameters List
N_assets = 2                        # from Parameters List
r_A = 0.5                           # net rate of return for asset A
r_B = 1.0                           # base coefficient for asset B (used with exponent)
exponent_B = 1.2                    # power for asset B return
lower_bound_A = 1.5                 # minimum allocation to asset A
lower_bound_B = 0.0                 # minimum allocation to asset B (no short selling)
risk_limit = 9                      # total risk limit
risk_power = 2                      # power used in risk measure (squared allocations)

# ===================================
# 2. Create model
# ===================================
model = gp.Model("BlueOcean_Asset_Allocation")

# Allow nonconvex features (due to general power constraints and nonlinear obj)
model.Params.NonConvex = 2

# ===================================
# 3. Create decision variables
# ===================================
# x_A: allocation to asset A
x_A = model.addVar(
    vtype=GRB.CONTINUOUS,
    lb=lower_bound_A,
    name="x_A"
)

# x_B: allocation to asset B
x_B = model.addVar(
    vtype=GRB.CONTINUOUS,
    lb=lower_bound_B,
    name="x_B"
)

# ==========================================
# 4. Auxiliary substitution variables
#    (range from -INF to +INF)
# ==========================================
# rA = x_A^2 (risk contribution from A)
rA = model.addVar(
    vtype=GRB.CONTINUOUS,
    lb=-GRB.INFINITY,
    ub=GRB.INFINITY,
    name="rA"
)

# rB = x_B^2 (risk contribution from B)
rB = model.addVar(
    vtype=GRB.CONTINUOUS,
    lb=-GRB.INFINITY,
    ub=GRB.INFINITY,
    name="rB"
)

# uB = x_B^1.2 (nonlinear return part from B)
uB = model.addVar(
    vtype=GRB.CONTINUOUS,
    lb=-GRB.INFINITY,
    ub=GRB.INFINITY,
    name="uB"
)

# ===================================
# 5. Link auxiliary variables
# ===================================
# rA = x_A^2
model.addGenConstrPow(x_A, rA, float(risk_power), name="riskA_pow")

# rB = x_B^2
model.addGenConstrPow(x_B, rB, float(risk_power), name="riskB_pow")

# uB = x_B^1.2
model.addGenConstrPow(x_B, uB, exponent_B, name="returnB_pow")

# ===================================
# 6. Add constraints
# ===================================

# (1) Minimum allocation to asset A is already enforced via lb of x_A (x_A >= 1.5)

# (2) No short selling of asset B is already enforced via lb of x_B (x_B >= 0)

# (3) Risk tolerance limit: x_A^2 + x_B^2 <= 9  ->  rA + rB <= risk_limit
model.addConstr(rA + rB <= risk_limit, name="RiskToleranceLimit")

# ===================================
# 7. Set objective function
# ===================================
# Maximize total return: 0.5 * x_A + x_B^1.2
# Using the auxiliary variable uB for x_B^1.2
model.setObjective(r_A * x_A + uB, GRB.MAXIMIZE)

# ===================================
# 8. Optimize model
# ===================================
model.optimize()

# ===================================
# 9. Print results
# ===================================
if model.Status == GRB.OPTIMAL:
    xA_val = x_A.X
    xB_val = x_B.X
    rA_val = rA.X
    rB_val = rB.X
    uB_val = uB.X
    obj_val = model.ObjVal

    print("Optimal solution found:")
    print(f"  x_A (allocation to asset A) = {xA_val:.6f}")
    print(f"  x_B (allocation to asset B) = {xB_val:.6f}")
    print(f"  rA = x_A^2                 = {rA_val:.6f}")
    print(f"  rB = x_B^2                 = {rB_val:.6f}")
    print(f"  uB = x_B^{exponent_B}         = {uB_val:.6f}")
    print(f"  Maximum return (objective) = {obj_val:.6f}")
else:
    print(f"Optimization ended with status {model.Status}")
    xA_val = float('nan')
    xB_val = float('nan')
    obj_val = float('nan')

# ===================================
# 10. Final answer output
# ===================================
# The question asks: "determine the capital allocation plan that can maximize
# investment returns and give the maximum return."
# We output the maximum return as the FinalAnswer value.
print(f"FinalAnswer=【{obj_val}】")