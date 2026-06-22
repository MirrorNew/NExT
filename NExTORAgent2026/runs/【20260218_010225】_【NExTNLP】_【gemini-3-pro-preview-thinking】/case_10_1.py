import gurobipy as gp
from gurobipy import GRB
import math

# 1. Create the model
model = gp.Model("HighPressureVesselDesign")

# 2. Define Parameters
inner_radius = 10
min_head_to_cyl_ratio = 5.0
min_cyl_to_length_ratio = 0.001
max_cylindrical_length = 240
min_total_volume = 70000
cost_material_per_unit = 1
welding_cost_coefficient = 2  # Used to verify objective multipliers

# 3. Create Decision Variables
# t_c: cylinder-wall thickness
t_c = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="t_c")
# t_h: head-wall thickness
t_h = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="t_h")
# L: length of cylindrical section
L = model.addVar(lb=0.0, ub=max_cylindrical_length, vtype=GRB.CONTINUOUS, name="L")

# 4. Create Auxiliary Variables for nonlinear terms
# We follow advice to set bounds to -infinity to +infinity for intermediate substitutions
# to avoid potential infeasibility during presolve, although physically they are positive.
aux_base_tc = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="aux_base_tc")
aux_sq_tc = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="aux_sq_tc")
aux_prod_L_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="aux_prod_L_sq")

aux_base_th = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="aux_base_th")
aux_cube_th = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="aux_cube_th")

# 5. Set NonConvex Parameter
model.Params.NonConvex = 2

# 6. Define Constraints

# 6.1 Auxiliary variable definitions
# aux_base_tc = r + t_c
model.addConstr(aux_base_tc == inner_radius + t_c, name="link_base_tc")

# aux_sq_tc = (r + t_c)^2
model.addGenConstrPow(aux_base_tc, aux_sq_tc, 2, name="pow_tc_2")

# aux_prod_L_sq = L * (r + t_c)^2
# Gurobi handles quadratic equality constraints with NonConvex=2
model.addConstr(aux_prod_L_sq == L * aux_sq_tc, name="link_prod_L_sq")

# aux_base_th = r + t_h
model.addConstr(aux_base_th == inner_radius + t_h, name="link_base_th")

# aux_cube_th = (r + t_h)^3
model.addGenConstrPow(aux_base_th, aux_cube_th, 3, name="pow_th_3")

# 6.2 Problem Specific Constraints

# Head-to-cylinder thickness ratio: t_h >= 5 * t_c
model.addConstr(t_h >= min_head_to_cyl_ratio * t_c, name="Head_Cyl_Ratio")

# Cylinder-thickness minimum bound: t_c >= 0.001 * L
model.addConstr(t_c >= min_cyl_to_length_ratio * L, name="Cyl_Thickness_Min")

# Storage-volume requirement (Inner volume)
# Volume = pi * r^2 * L + (4/3) * pi * r^3
cyl_vol = math.pi * (inner_radius**2) * L
head_vol = (4.0/3.0) * math.pi * (inner_radius**3)
model.addConstr(cyl_vol + head_vol >= min_total_volume, name="Min_Volume")

# Cylinder length upper bound (already set in variable definition, but can add explicit constraint)
model.addConstr(L <= max_cylindrical_length, name="Max_Length")

# 7. Set Objective Function
# Objective: Minimize Manufacturing Cost
# Cost = Material Cost + Welding Cost
# Material Cost = 1 * (Volume_Cyl_Material + Volume_Head_Material)
# Volume_Cyl_Material = pi * L * ((r+t_c)^2 - r^2) = pi * (L * (r+t_c)^2 - L*r^2) = pi * (aux_prod_L_sq - L*r^2)
# Volume_Head_Material = (4/3) * pi * ((r+t_h)^3 - r^3) = (4/3) * pi * (aux_cube_th - r^3)

material_cost_cyl = math.pi * (aux_prod_L_sq - (inner_radius**2) * L)
material_cost_head = (4.0/3.0) * math.pi * (aux_cube_th - (inner_radius**3))
material_cost = cost_material_per_unit * (material_cost_cyl + material_cost_head)

# Welding Cost = 4 * L + 8 * pi * (r + t_c)
# Note: The problem states "welding cost to be twice the welding length".
# Longitudinal length (approx) = 2 * L (inner+outer), cost = 2 * 2L = 4L
# Circumferential length = 2 * (2 * pi * (r+t_c)), cost = 2 * Length = 8 * pi * (r+t_c)
welding_cost = 4 * L + 8 * math.pi * aux_base_tc

total_cost = material_cost + welding_cost

model.setObjective(total_cost, GRB.MINIMIZE)

# 8. Solve the model
model.optimize()

# 9. Print Results
if model.status == GRB.OPTIMAL:
    print(f"Optimal Cost Found: {model.ObjVal}")
    print(f"Variables: L={L.X}, t_c={t_c.X}, t_h={t_h.X}")
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Model did not solve to optimality.")
    if model.status == GRB.INFEASIBLE:
         print("Model is infeasible.")
    print(f"FinalAnswer=【None】")