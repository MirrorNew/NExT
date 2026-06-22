import gurobipy as gp
from gurobipy import GRB
import math

# ===========================
# 1. Define parameters
# ===========================
inner_radius = 10                           # r
min_head_to_cyl_ratio = 5.0                # t_h >= 5 * t_c
min_cyl_to_length_ratio = 0.001            # t_c >= 0.001 * L
max_cylindrical_length = 240               # L <= 240
min_total_volume = 70000                   # volume >= 70000
cost_material_per_unit = 1                 # not needed explicitly (already in objective)
welding_cost_coefficient = 2               # used in welding cost = 2 * length

pi = math.pi
r = inner_radius

# ===========================
# 2. Create model
# ===========================
model = gp.Model("HighPressureVesselDesign")

# Allow nonconvex quadratic/cubic expressions via general constraints
model.Params.NonConvex = 2

# ===========================
# 3. Decision variables
# ===========================
t_c = model.addVar(lb=0.0, name="t_c")                          # cylinder-wall thickness
t_h = model.addVar(lb=0.0, name="t_h")                          # head-wall thickness
L   = model.addVar(lb=0.0, ub=max_cylindrical_length, name="L") # cylindrical length

# ===========================
# 4. Auxiliary variables
# ===========================
# For powers involving decision variables:
# z1 = (r + t_c)^2
# z2 = (r + t_h)^3
# Auxiliary cost variables to keep objective linear in them:
# material_cyl_cost = π * ((r + t_c)^2 - r^2) * L
# material_head_cost = (4/3) * π * ((r + t_h)^3 - r^3)

z1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="z1")  # (r + t_c)^2
z2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="z2")  # (r + t_h)^3

material_cyl_cost = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="material_cyl_cost")
material_head_cost = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="material_head_cost")

# ===========================
# 5. General constraints for powers and cost definitions
# ===========================
# Implement z1 = (r + t_c)^2 and z2 = (r + t_h)^3
# Introduce intermediate variables s1 = r + t_c, s2 = r + t_h so that we can use GenConstrPow
s1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="s1")  # r + t_c
s2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="s2")  # r + t_h

model.addConstr(s1 == r + t_c, name="def_s1")
model.addConstr(s2 == r + t_h, name="def_s2")

# z1 = s1^2, z2 = s2^3
model.addGenConstrPow(s1, z1, 2.0, name="pow_z1")
model.addGenConstrPow(s2, z2, 3.0, name="pow_z2")

# Define the material cost components as linear expressions in z1, z2, and L
# material_cyl_cost = π * (z1 - r^2) * L
# material_head_cost = (4/3) * π * (z2 - r^3)
# These are still nonlinear because of multiplication by L, but we keep them as
# variables linked through algebraic constraints.
# Introduce auxiliary term (z1 - r^2) as w1 and (z2 - r^3) as w2 to keep the expressions clear.
w1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="w1")  # z1 - r^2
w2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="w2")  # z2 - r^3

model.addConstr(w1 == z1 - r * r, name="def_w1")
model.addConstr(w2 == z2 - r * r * r, name="def_w2")

# Now connect material_cyl_cost and material_head_cost
model.addConstr(material_cyl_cost == pi * w1 * L, name="def_material_cyl_cost")
model.addConstr(material_head_cost == (4.0 / 3.0) * pi * w2, name="def_material_head_cost")

# ===========================
# 6. Constraints
# ===========================
# 1) Head-to-cylinder thickness ratio
model.addConstr(t_h >= min_head_to_cyl_ratio * t_c, name="head_to_cyl_ratio")

# 2) Cylinder-thickness minimum bound: t_c >= 0.001 * L
model.addConstr(t_c >= min_cyl_to_length_ratio * L, name="cyl_thickness_min")

# 3) Cylinder length upper bound: L <= 240 (already via variable bound, but add explicit constraint)
model.addConstr(L <= max_cylindrical_length, name="length_upper_bound")

# 4) Storage-volume requirement:
# π r^2 L + (4/3) π r^3 >= 70000
model.addConstr(pi * r * r * L + (4.0 / 3.0) * pi * r * r * r >= min_total_volume,
                name="volume_requirement")

# ===========================
# 7. Objective function
# ===========================
# Total manufacturing cost:
# C_total = material_cyl_cost
#         + material_head_cost
#         + welding_cost_longitudinal
#         + welding_cost_circumferential
#
# Given from problem:
# welding_cost_longitudinal = 4 * L
# welding_cost_circumferential = 8 * π * (r + t_c)
# welding_cost_coefficient = 2 already embedded in these expressions in the given model.
#
# So:
# Minimize C_total = material_cyl_cost
#                  + material_head_cost
#                  + 4 * L
#                  + 8 * π * (r + t_c)

C_total = material_cyl_cost + material_head_cost + 4.0 * L + 8.0 * pi * (r + t_c)
model.setObjective(C_total, GRB.MINIMIZE)

# ===========================
# 8. Optimize
# ===========================
model.optimize()

# ===========================
# 9. Print results
# ===========================
if model.Status == GRB.OPTIMAL:
    t_c_opt = t_c.X
    t_h_opt = t_h.X
    L_opt = L.X
    C_opt = C_total.getValue()

    print(f"Optimal cylinder-wall thickness t_c = {t_c_opt}")
    print(f"Optimal head-wall thickness t_h = {t_h_opt}")
    print(f"Optimal cylindrical length L = {L_opt}")
    print(f"Minimum manufacturing cost C_total = {C_opt}")

    # According to the problem statement, the question asks:
    # "Design the container to minimize the manufacturing cost and give the minimum cost."
    # So the final answer is the minimum cost.
    print(f"FinalAnswer=【{C_opt}】")
else:
    # If not optimal, still output something for FinalAnswer (e.g., NaN)
    print("No optimal solution found.")
    print("FinalAnswer=【nan】")