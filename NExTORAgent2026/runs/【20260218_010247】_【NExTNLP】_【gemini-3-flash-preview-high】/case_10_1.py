import gurobipy as gp
from gurobipy import GRB
import math

# Define all parameter values from the provided Parameters List
inner_radius = 10
min_head_to_cyl_ratio = 5.0
min_cyl_to_length_ratio = 0.001
max_cylindrical_length = 240
min_total_volume = 70000
cost_material_per_unit = 1
welding_cost_coefficient = 2

# Create the Gurobi model
model = gp.Model("VesselOptimization")

# Create decision variables
t_c = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t_c")
t_h = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t_h")
L = model.addVar(lb=0, ub=max_cylindrical_length, vtype=GRB.CONTINUOUS, name="L")

# Create auxiliary substitution variables for nonlinear terms
# According to instructions, auxiliary variables should have range from -infinity to +infinity
t_sq_c = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t_sq_c")
t_sq_h = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t_sq_h")
t_cu_h = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t_cu_h")
t_cL = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t_cL")
t_sqcL = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t_sqcL")

# Set the NonConvex parameter to 2 to allow for non-convex terms like bilinear and higher order terms
model.Params.NonConvex = 2

# Define constraints to link decision variables to auxiliary variables
# Link t_sq_c = t_c^2
model.addGenConstrPow(t_c, t_sq_c, 2.0)
# Link t_sq_h = t_h^2
model.addGenConstrPow(t_h, t_sq_h, 2.0)
# Link t_cu_h = t_h^3
model.addGenConstrPow(t_h, t_cu_h, 3.0)
# Link t_cL = t_c * L
model.addConstr(t_cL == t_c * L)
# Link t_sqcL = t_sq_c * L
model.addConstr(t_sqcL == t_sq_c * L)

# Set up the objective function based on the validated mathematical model
# Material Volume Cylinder = pi * (2 * r * t_c + t_c^2) * L = 20 * pi * t_c * L + pi * t_c^2 * L
# Material Volume Heads = (4/3) * pi * (3 * r^2 * t_h + 3 * r * t_h^2 + t_h^3) = 400 * pi * t_h + 40 * pi * t_h^2 + (4/3) * pi * t_h^3
# Welding Cost = 4 * L + 8 * pi * (r + t_c) = 4 * L + 80 * pi + 8 * pi * t_c
# Total Cost = Cylinder Material + Head Material + Welding Cost
objective = (20 * math.pi * t_cL + math.pi * t_sqcL + 
             400 * math.pi * t_h + 40 * math.pi * t_sq_h + (4/3) * math.pi * t_cu_h + 
             4 * L + 8 * math.pi * t_c + 80 * math.pi)
model.setObjective(objective, GRB.MINIMIZE)

# Add constraints
# 1. Head-to-cylinder thickness ratio: t_h >= 5 * t_c
model.addConstr(t_h >= min_head_to_cyl_ratio * t_c, "head_to_cylinder_ratio")

# 2. Cylinder-thickness minimum bound: t_c >= L / 1000
model.addConstr(t_c >= min_cyl_to_length_ratio * L, "cylinder_thickness_min")

# 3. Cylinder length upper bound (already handled in variable definition)
model.addConstr(L <= max_cylindrical_length, "max_cylinder_length")

# 4. Storage-volume requirement: pi * r^2 * L + (4/3) * pi * r^3 >= 70000
model.addConstr(math.pi * (inner_radius**2) * L + (4/3) * math.pi * (inner_radius**3) >= min_total_volume, "storage_volume_requirement")

# Solve the model
model.optimize()

# Print the results and final objective value
if model.status == GRB.OPTIMAL:
    print(f"Minimum manufacturing cost: {model.objVal}")
    print(f"Cylinder-wall thickness: {t_c.X}")
    print(f"Head-wall thickness: {t_h.X}")
    print(f"Length of cylindrical section: {L.X}")
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("Optimization was not successful.")