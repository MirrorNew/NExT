import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Define all parameters
# =========================
h_ideal = 100.0
h_threshold = 90.0
h_min_downstream = 10.0
V_storage = 100000000.0
h_initial = 100.0
drop_rate = 1.25
gen_coefficient = 5.0
consumption_coeff = 0.5

# Note: problem is modeled in "million m^3" for V to match given ranges: 0 ≤ V ≤ 100

# =========================
# 2. Create model
# =========================
model = gp.Model("Reservoir_Water_Energy_Optimization")

# Allow non-convex quadratic / bilinear terms
model.Params.NonConvex = 2

# =========================
# 3. Create decision variables
# =========================
# V: water release volume (million m^3)
V = model.addVar(lb=0.0, ub=100.0, vtype=GRB.CONTINUOUS, name="V")

# h: reservoir head height after release (m)
h = model.addVar(lb=10.0, ub=100.0, vtype=GRB.CONTINUOUS, name="h")

# P_u: unit water power generation
P_u = model.addVar(lb=275.0, ub=500.0, vtype=GRB.CONTINUOUS, name="P_u")

# P: total power generation
P = model.addVar(lb=0.0, ub=50000.0, vtype=GRB.CONTINUOUS, name="P")

# Δ⁺: auxiliary variable for (90 - h)^+
Delta_plus = model.addVar(lb=0.0, ub=80.0, vtype=GRB.CONTINUOUS, name="Delta_plus")

# C: additional energy consumption penalty
C = model.addVar(lb=0.0, ub=3200.0, vtype=GRB.CONTINUOUS, name="C")

# =========================
# 4. Auxiliary substitution variables
# =========================
# C_sq: represents (Delta_plus)^2
C_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS, name="C_sq")

# =========================
# 5. Objective function
# =========================
# maximize Z = P - C
model.setObjective(P - C, GRB.MAXIMIZE)

# =========================
# 6. Add constraints
# =========================

# 6.1 Volume capacity (already enforced by bounds on V, but we keep explicit if desired)
model.addConstr(V >= 0.0, name="V_min")
model.addConstr(V <= 100.0, name="V_max")

# 6.2 Head–volume relation: h = 100 – 1.25·V
model.addConstr(h == h_initial - drop_rate * V, name="Head_Volume")

# 6.3 Minimum head requirement: h ≥ 10
model.addConstr(h >= h_min_downstream, name="Min_Head")

# 6.4 Unit power generation definition: P_u = 5·(100 + h)/2
model.addConstr(P_u == gen_coefficient * (h_ideal + h) / 2.0,
                name="Unit_Power")

# 6.5 Total power generation: P = P_u·V  (bilinear)
model.addConstr(P == P_u * V, name="Total_Power")

# 6.6 Penalty auxiliary nonnegativity: Delta_plus ≥ 0
model.addConstr(Delta_plus >= 0.0, name="Delta_nonneg")

# 6.7 Penalty auxiliary threshold: Delta_plus ≥ 90 – h
model.addConstr(Delta_plus >= h_threshold - h, name="Delta_threshold")

# 6.8 Additional consumption definition: C = 0.5·(Delta_plus)²
# First: C_sq = (Delta_plus)^2 via general power constraint
model.addGenConstrPow(Delta_plus, C_sq, 2.0, name="Delta_square")

# Then: C = 0.5 * C_sq
model.addConstr(C == consumption_coeff * C_sq, name="Penalty_definition")

# =========================
# 7. Solve the model
# =========================
model.optimize()

# =========================
# 8. Print results
# =========================
if model.status == GRB.OPTIMAL:
    V_opt = V.X
    h_opt = h.X
    Pu_opt = P_u.X
    P_opt = P.X
    Delta_opt = Delta_plus.X
    C_opt = C.X
    Z_opt = model.ObjVal

    print("Optimal solution found:")
    print(f"  Water release V (million m^3): {V_opt:.6f}")
    print(f"  Head after release h (m):      {h_opt:.6f}")
    print(f"  Unit generation P_u:           {Pu_opt:.6f}")
    print(f"  Total generation P:            {P_opt:.6f}")
    print(f"  Delta_plus (90 - h)^+:         {Delta_opt:.6f}")
    print(f"  Penalty C:                     {C_opt:.6f}")
    print(f"  Objective Z = P - C:           {Z_opt:.6f}")

    # According to the problem, the "question answer" is:
    # "the water release decision that maximizes the power generation revenue"
    # Here we output the optimal water release volume V.
    print(f"FinalAnswer=【{V_opt}】")
else:
    print("No optimal solution found.")
    # Still output something for FinalAnswer to conform to required format
    print("FinalAnswer=【None】")