import gurobipy as gp
from gurobipy import GRB
import math

# 1. Import Gurobi and any other necessary packages.
# math is imported for pi.

# 2. Define all parameter matrices and data inputs.
G = 11500000.0
rho = 0.284
F_work = 300
tau_allowable = 80000
f_max = 4
delta_allowable = 0.5
outer_diameter_max = 1.5
d_min, d_max = 0.1, 0.5
D_min, D_max = 0.5, 6.0
weight_constant = 1
pi = math.pi

# 3. Create decision variables.
model = gp.Model("SpringDesign")
d = model.addVar(lb=d_min, ub=d_max, name="d")
D = model.addVar(lb=D_min, ub=D_max, name="D")
N = model.addVar(lb=1, vtype=GRB.INTEGER, name="N")

# 4. Create any auxiliary substitution or indicator variables.
d_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="d_sq")
d_cub = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="d_cub")
d_p4 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="d_p4")
D_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="D_sq")
D_cub = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="D_cub")
D_p4 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="D_p4")
Dd = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Dd")
d3D = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="d3D")
N_plus_2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="N_plus_2")
N2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="N2")
D_p4_N2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="D_p4_N2")
D_cub_N = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="D_cub_N")
d2D = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="d2D")
d2DN2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="d2DN2")

# General constraints for powers
model.addGenConstrPow(d, d_sq, 2.0)
model.addGenConstrPow(d, d_cub, 3.0)
model.addGenConstrPow(d, d_p4, 4.0)
model.addGenConstrPow(D, D_sq, 2.0)
model.addGenConstrPow(D, D_cub, 3.0)
model.addGenConstrPow(D, D_p4, 4.0)

# Constraints for products
model.addConstr(Dd == D * d)
model.addConstr(d3D == d_cub * D)
model.addConstr(N_plus_2 == N + 2)
model.addConstr(N2 == N * N_plus_2)
model.addConstr(D_p4_N2 == D_p4 * N2)
model.addConstr(D_cub_N == D_cub * N)
model.addConstr(d2D == d_sq * D)
model.addConstr(d2DN2 == d2D * N_plus_2)

# 5. Set up the objective function.
model.setObjective(weight_constant * (pi**2 / 4) * d2DN2, GRB.MINIMIZE)

# 6. Add all constraints.
# Shear stress constraint: simplified from tau <= tau_allowable
# After rearranging: 8*F_work*D^2 + (8*0.615*F_work - 2*F_work)*D*d - 8*0.615*F_work*d^2 <= tau_allowable*pi*(d^3*D - d^4)
# 8*300 = 2400; 8*0.615*300 = 1476; 1476 - 2*300 = 876.
model.addConstr(2400 * D_sq + 876 * Dd - 1476 * d_sq <= tau_allowable * pi * (d3D - d_p4))

# Natural frequency constraint: simplified from f <= f_max
# G * d^2 <= 128 * rho * pi^4 * D^4 * N * (N+2)
model.addConstr(G * d_sq <= 128 * rho * (pi**4) * D_p4_N2)

# Minimum allowable compression: simplified from delta_max >= delta_allowable
# 8 * F_work * D^3 * N >= delta_allowable * G * d^4
model.addConstr(8 * F_work * D_cub_N >= delta_allowable * G * d_p4)

# Winding ratio constraint: 4 <= D/d <= 12
model.addConstr(D >= 4 * d)
model.addConstr(D <= 12 * d)

# Outer diameter constraint: D + d <= 1.5
model.addConstr(D + d <= outer_diameter_max)

# 7. Solve the model and print results.
model.Params.NonConvex = 2
model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("Optimization was not successful.")