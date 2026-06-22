import gurobipy as gp
from gurobipy import GRB

# Define parameters from Parameters List
M_req = 200.0
b_range = [200.0, 1000.0]
h_range = [200.0, 1000.0]
A_s_max = 20000.0
f_c_prime = 30.0
f_y = 400.0
phi = 0.9
rho_min = 0.035
k = 0.85
cost_area_coeff = 1.0
steel_concrete_cost_ratio = 15.0

# Create model
model = gp.Model("BeamDesignOptimization")
model.Params.NonConvex = 2  # For bilinear terms

# Decision variables
b = model.addVar(lb=b_range[0], ub=b_range[1], name="b")
h = model.addVar(lb=h_range[0], ub=h_range[1], name="h")
A_s = model.addVar(lb=0.0, ub=A_s_max, name="A_s")
a = model.addVar(lb=0.0, ub=GRB.INFINITY, name="a")
z = model.addVar(lb=0.0, ub=GRB.INFINITY, name="z")
M_n = model.addVar(lb=M_req, ub=GRB.INFINITY, name="M_n")
rho = model.addVar(lb=rho_min, ub=GRB.INFINITY, name="rho")

# Auxiliary substitution variables
Y1 = model.addVar(lb=0.0, ub=GRB.INFINITY, name="Y1")  # Y1 = b*h
Y2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Y2")  # Y2 = 1/(b*h)
Y3 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Y3")  # Y3 = (A_s*f_y)/(k*f_c'*b)
Y4 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Y4")  # Y4 = 1/b

# Constraints for auxiliary variables
model.addConstr(Y1 == b * h, name="Y1_def")

# For Y2 = 1/(b*h), we use b*h * Y2 == 1
model.addConstr(Y1 * Y2 == 1, name="Y2_reciprocal")

# For Y4 = 1/b, we use b * Y4 == 1
model.addConstr(b * Y4 == 1, name="Y4_reciprocal")

# For Y3 = (A_s*f_y)/(k*f_c'*b) = (A_s*f_y)/(k*f_c') * Y4
model.addConstr(Y3 == (A_s * f_y) / (k * f_c_prime) * Y4, name="Y3_def")

# Force equilibrium: A_s*f_y = k*f_c'*b*a  ->  a = (A_s*f_y)/(k*f_c'*b) = Y3
model.addConstr(a == Y3, name="force_equilibrium")

# Lever arm definition (z = h - a/2)
model.addConstr(z == h - a / 2, name="z_def")

# Alternative z definition for consistency: z = h - Y3/2
model.addConstr(z == h - Y3 / 2, name="z_def_alternative")

# Bending capacity definition: M_n = phi * A_s * f_y * z
model.addConstr(M_n == phi * A_s * f_y * z, name="Mn_definition")

# Reinforcement ratio definition: rho = A_s/(b*h) = A_s * Y2
model.addConstr(rho == A_s * Y2, name="rho_definition")

# Minimum reinforcement ratio constraint (enforced by variable bounds on rho)

# Objective function: Minimize C = b*h + 15*A_s = Y1 + 15*A_s
C = model.addVar(lb=0.0, ub=GRB.INFINITY, name="C")
model.addConstr(C == Y1 + steel_concrete_cost_ratio * A_s, name="cost_def")
model.setObjective(C, GRB.MINIMIZE)

# Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print(f"Optimal solution found")
    print(f"b = {b.X:.2f} mm")
    print(f"h = {h.X:.2f} mm")
    print(f"A_s = {A_s.X:.2f} mm²")
    print(f"a = {a.X:.2f} mm")
    print(f"z = {z.X:.2f} mm")
    print(f"M_n = {M_n.X:.2f} kN·m")
    print(f"ρ = {rho.X:.4f}")
    print(f"Minimum cost C = {C.X:.2f}")
    
    # Output final answer as required
    print(f"FinalAnswer=【{C.X:.2f}】")
else:
    print(f"Optimization failed with status {model.status}")
    print(f"FinalAnswer=【None】")