import gurobipy as gp
from gurobipy import GRB

# ==============================
# 1. Define parameters (from Parameters List)
# ==============================
M_req = 200.0                      # kN·m
concrete_grade = "C30"
steel_grade = "HRB400"
b_range = [200.0, 1000.0]          # mm
h_range = [200.0, 1000.0]          # mm
A_s_max = 20000.0                  # mm^2
f_c_prime = 30.0                   # MPa
f_y = 400.0                        # MPa
phi = 0.9
rho_min = 0.035
k = 0.85
cost_area_coeff = 1.0              # for concrete bh term
steel_concrete_cost_ratio = 15.0   # steel vs concrete cost

# ==============================
# 2. Create model
# ==============================
model = gp.Model("RC_Beam_Section_Design")
model.Params.NonConvex = 2  # Needed for bilinear terms

# ==============================
# 3. Decision variables
# ==============================
# Geometry and steel
b = model.addVar(lb=b_range[0], ub=b_range[1], vtype=GRB.CONTINUOUS, name="b")
h = model.addVar(lb=h_range[0], ub=h_range[1], vtype=GRB.CONTINUOUS, name="h")
A_s = model.addVar(lb=0.0, ub=A_s_max, vtype=GRB.CONTINUOUS, name="A_s")

# Stress block and lever arm
a = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="a")
z = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="z")

# Nominal bending capacity
M_n = model.addVar(lb=M_req, vtype=GRB.CONTINUOUS, name="M_n")

# Reinforcement ratio
rho = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="rho")

# ==============================
# 4. Auxiliary substitution variables (bilinear products)
#    All with full range (-inf, inf) as required
# ==============================
t_bh = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS, name="t_bh")           # b * h
t_As_fy = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                       vtype=GRB.CONTINUOUS, name="t_As_fy")     # A_s * f_y
t_b_a = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                     vtype=GRB.CONTINUOUS, name="t_b_a")         # b * a
t_As_fy_z = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                         vtype=GRB.CONTINUOUS, name="t_As_fy_z") # (A_s * f_y) * z

# (Optional, as suggested text: A_s * h; not strictly needed in constraints, but created)
t_As_h = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                      vtype=GRB.CONTINUOUS, name="t_As_h")       # A_s * h

# ==============================
# 5. Objective function
#     Cost = cost_area_coeff * (b*h) + steel_concrete_cost_ratio * A_s
#     Implemented via t_bh = b * h
# ==============================
model.setObjective(cost_area_coeff * t_bh +
                   steel_concrete_cost_ratio * A_s,
                   sense=GRB.MINIMIZE)

# ==============================
# 6. Constraints
# ==============================

# 6.1 Bilinear definitions
# t_bh = b * h
model.addConstr(t_bh == b * h, name="def_t_bh")

# t_As_h = A_s * h (not used elsewhere but included as per advice)
model.addConstr(t_As_h == A_s * h, name="def_t_As_h")

# t_As_fy = A_s * f_y
model.addConstr(t_As_fy == A_s * f_y, name="def_t_As_fy")

# t_b_a = b * a
model.addConstr(t_b_a == b * a, name="def_t_b_a")

# t_As_fy_z = (A_s * f_y) * z
model.addConstr(t_As_fy_z == t_As_fy * z, name="def_t_As_fy_z")

# 6.2 Reinforcement ratio relation: A_s = rho * (b * h) using t_bh
model.addConstr(A_s == rho * t_bh, name="rho_definition")

# Minimum reinforcement ratio: rho >= rho_min
model.addConstr(rho >= rho_min, name="rho_minimum")

# 6.3 Force equilibrium: A_s * f_y = k * f_c_prime * b * a
# Using t_As_fy and t_b_a
model.addConstr(t_As_fy == k * f_c_prime * t_b_a, name="force_equilibrium")

# 6.4 Lever arm definition: z = h - a/2
model.addConstr(z == h - 0.5 * a, name="z_def_rectangular")

# 6.5 Nominal bending capacity: M_n = phi * (A_s * f_y * z)
# Using t_As_fy_z
model.addConstr(M_n == phi * t_As_fy_z, name="Mn_definition")

# 6.6 Moment capacity requirement: M_n >= M_req
model.addConstr(M_n >= M_req, name="moment_capacity")

# Bounds on b, h, A_s are already built into variable definitions

# ==============================
# 7. Solve model and print results
# ==============================
model.optimize()

if model.status == GRB.OPTIMAL:
    b_val = b.X
    h_val = h.X
    A_s_val = A_s.X
    a_val = a.X
    z_val = z.X
    M_n_val = M_n.X
    rho_val = rho.X
    cost_val = model.ObjVal

    print("Optimal solution found:")
    print(f"  b    = {b_val:.4f}")
    print(f"  h    = {h_val:.4f}")
    print(f"  A_s  = {A_s_val:.4f}")
    print(f"  a    = {a_val:.4f}")
    print(f"  z    = {z_val:.4f}")
    print(f"  M_n  = {M_n_val:.4f}")
    print(f"  rho  = {rho_val:.6f}")
    print(f"  Cost = {cost_val:.4f}")
else:
    print(f"Optimization ended with status {model.status}")
    cost_val = float('nan')

# ==============================
# 8. Final answer output
#    The question asks: "minimize the cross-sectional cost and give the calculated cost value."
#    So FinalAnswer is the optimal cost.
# ==============================
print(f"FinalAnswer=【{cost_val}】")