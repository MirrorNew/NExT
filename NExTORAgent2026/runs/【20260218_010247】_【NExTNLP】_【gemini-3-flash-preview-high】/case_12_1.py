import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and any other necessary packages.
# Already imported above.

# 2. Define all parameter matrices and data inputs.
M_req_value = 200  # kN·m
M_req = M_req_value * 1e6  # Convert to N·mm for consistency with MPa (N/mm²)
f_c_prime = 30.0  # MPa
f_y = 400.0  # MPa
phi = 0.9
rho_min = 0.035
k = 0.85
b_min, b_max = 200.0, 1000.0
h_min, h_max = 200.0, 1000.0
As_max = 20000.0
steel_concrete_cost_ratio = 15.0

# 3. Create decision variables.
model = gp.Model("BeamOptimization")
model.Params.NonConvex = 2  # Enable non-convex optimization

b = model.addVar(lb=b_min, ub=b_max, name="b")
h = model.addVar(lb=h_min, ub=h_max, name="h")
As = model.addVar(lb=0, ub=As_max, name="As")

# 4. Create auxiliary substitution variables.
# Range from negative infinity to positive infinity as per instructions.
bh = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="bh")
ba = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="ba")
Asz = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Asz")

# Intermediate structural variables
a = model.addVar(lb=0, ub=GRB.INFINITY, name="a")
z = model.addVar(lb=0, ub=GRB.INFINITY, name="z")
Mn = model.addVar(lb=0, ub=GRB.INFINITY, name="Mn")

# 5. Set up the objective function.
# C(b,h,As) = b*h + 15*As
model.setObjective(bh + steel_concrete_cost_ratio * As, GRB.MINIMIZE)

# 6. Add all constraints (including gen-constr and indicator constraints).
# Definition of auxiliary variables to handle bilinear terms and eliminate denominators
model.addConstr(bh == b * h, name="bh_definition")
model.addConstr(ba == b * a, name="ba_definition")
model.addConstr(Asz == As * z, name="Asz_definition")

# Force equilibrium: As * fy = k * fc' * b * a
# 400 * As = 0.85 * 30 * ba = 25.5 * ba
model.addConstr(f_y * As == k * f_c_prime * ba, name="force_equilibrium")

# Internal lever arm calculation: z = h - a/2
model.addConstr(z == h - 0.5 * a, name="internal_arm_definition")

# Bending capacity calculation: Mn = phi * As * fy * z
# Mn = 0.9 * 400 * Asz = 360 * Asz
model.addConstr(Mn == phi * f_y * Asz, name="bending_capacity_definition")

# Design requirement: Mn >= M_req
model.addConstr(Mn >= M_req, name="capacity_requirement")

# Minimum reinforcement ratio requirement: rho = As / (b * h) >= rho_min
model.addConstr(As >= rho_min * bh, name="min_reinforcement_ratio")

# 7. Solve the model and print results.
model.optimize()

if model.status == GRB.OPTIMAL:
    objective_value = model.objVal
    print(f"Optimal width (b): {b.X}")
    print(f"Optimal height (h): {h.X}")
    print(f"Optimal steel area (As): {As.X}")
    print(f"FinalAnswer=【{objective_value}】")
else:
    print("Optimal solution not found.")