import gurobipy as gp
import math

# Parameters from Parameters List
r = 10  # inner_radius
min_head_to_cyl_ratio = 5.0  # min_head_to_cyl_ratio
min_cyl_to_length_ratio = 0.001  # min_cyl_to_length_ratio
max_cyl_length = 240  # max_cylindrical_length
min_volume = 70000  # min_total_volume
cost_material_per_unit = 1  # cost_material_per_unit
welding_cost_coeff = 2  # welding_cost_coefficient

# Create model
model = gp.Model("HighPressureVesselDesign")

# Enable non-convex optimization for power constraints
model.Params.NonConvex = 2

# Decision variables
t_c = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="t_c")
t_h = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="t_h")
L = model.addVar(lb=0, ub=max_cyl_length, vtype=gp.GRB.CONTINUOUS, name="L")

# Auxiliary variables for nonlinear terms
Y1 = model.addVar(lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="Y1")  # (r + t_c)^2
Y2 = model.addVar(lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="Y2")  # (r + t_h)^3

# Power constraints - define Y1 = (r + t_c)^2
model.addGenConstrPow(r + t_c, Y1, 2, "pow_constr1")
# Define Y2 = (r + t_h)^3
model.addGenConstrPow(r + t_h, Y2, 3, "pow_constr2")

# Volume constraint: π·r²·L + (4/3)·π·r³ ≥ 70000
# With r=10: π*100*L + (4/3)*π*1000 ≥ 70000
# Corrected volume expression: 100*π*L + (4000/3)*π
volume_expr = 100 * math.pi * L + (4000/3) * math.pi
model.addConstr(volume_expr >= min_volume, "volume_constraint")

# Thickness ratio constraints
model.addConstr(t_h >= min_head_to_cyl_ratio * t_c, "head_cyl_ratio")
model.addConstr(t_c >= min_cyl_to_length_ratio * L, "cyl_length_ratio")

# Objective function
# Material cost: π*((r+t_c)^2 - r^2)*L + (4/3)*π*((r+t_h)^3 - r^3)
# = π*(Y1 - 100)*L + (4/3)*π*(Y2 - 1000)
material_cost = math.pi * (Y1 - 100) * L + (4/3) * math.pi * (Y2 - 1000)

# Welding cost: 4*L + 8*π*(r+t_c)
welding_cost = 4 * L + 8 * math.pi * (r + t_c)

# Total cost
total_cost = material_cost + welding_cost

# Set objective
model.setObjective(total_cost, gp.GRB.MINIMIZE)

# Solve
model.optimize()

# Print results
print("\n--- Optimal Solution ---")
print(f"Cylinder thickness (t_c): {t_c.X:.6f}")
print(f"Head thickness (t_h): {t_h.X:.6f}")
print(f"Cylinder length (L): {L.X:.6f}")
print(f"Minimum total cost: {model.ObjVal:.6f}")

# Final answer: minimum cost
print(f"FinalAnswer=【{model.ObjVal:.6f}】")