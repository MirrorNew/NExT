import gurobipy as gp
from gurobipy import GRB
import math

# Define the model
model = gp.Model("Spring_Design_Optimization")

# Parameters
tolerance = 0.05
G = 11500000.0
rho = 0.284
F_work = 300
tau_allowable = 80000
f_max = 4
delta_allowable = 0.5
C_range = [4, 12]
outer_diameter_max = 1.5
d_range = [0.1, 0.5]
D_range = [0.5, 6.0]
weight_constant = 1

# Constants
PI = math.pi

# Set NonConvex parameter to 2 to handle quadratic equality constraints (e.g. z = x*y)
model.Params.NonConvex = 2

# 3. Create Decision Variables
d = model.addVar(lb=d_range[0], ub=d_range[1], vtype=GRB.CONTINUOUS, name="d")
D = model.addVar(lb=D_range[0], ub=D_range[1], vtype=GRB.CONTINUOUS, name="D")
N = model.addVar(lb=1, vtype=GRB.INTEGER, name="N")

# 4. Auxiliary Variables
# Powers of d
d_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="d_sq")
d_cub = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="d_cub")
d_quad = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="d_quad")

# Powers of D
D_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="D_sq")
D_cub = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="D_cub")
D_quad = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="D_quad")

# Powers of N
N_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="N_sq")

# Power constraints (GenConstrPow: y = x^a)
model.addGenConstrPow(d, d_sq, 2)
model.addGenConstrPow(d, d_cub, 3)
model.addGenConstrPow(d, d_quad, 4)

model.addGenConstrPow(D, D_sq, 2)
model.addGenConstrPow(D, D_cub, 3)
model.addGenConstrPow(D, D_quad, 4)

model.addGenConstrPow(N, N_sq, 2)

# Intermediate variables for Substitutions
# For Stress: stress_frac = (4D^2 - Dd) / (4D - 4d)
stress_frac = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="stress_frac")

# For Frequency: freq_term = D^4 * (N^2 + 2N)
freq_term = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="freq_term")

# For Deflection: def_term = D^3 * N
def_term = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="def_term")

# For Objective: obj_term_1 = d^2 * D, obj_term_2 = obj_term_1 * N
obj_term_1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="obj_term_1")
obj_term_2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="obj_term_2")

# 5. Objective Function
# Weight W = (pi^2/4) * d^2 * D * (N+2) = (pi^2/4) * (d^2*D*N + 2*d^2*D)
# W = (pi^2/4) * (obj_term_2 + 2 * obj_term_1)
model.setObjective((PI**2 / 4.0) * (obj_term_2 + 2 * obj_term_1) * weight_constant, GRB.MINIMIZE)

# 6. Constraints

# 6.1 Auxiliary Definitions (Bilinear/Quadratic Equalities)
# Stress fraction: stress_frac * (4D - 4d) == 4D^2 - D*d
model.addConstr(stress_frac * (4*D - 4*d) == 4*D_sq - D*d, name="Def_Stress_Frac")

# Frequency term: freq_term == D^4 * (N^2 + 2N)
# Note: D_quad * (N_sq + 2*N) is degree 6 product? 
# Gurobi GenConstrPow only does y=x^a. 
# We need to break down freq_term = D_quad * N_poly.
N_poly = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.INTEGER, name="N_poly")
model.addConstr(N_poly == N_sq + 2*N, name="Def_N_poly")
model.addConstr(freq_term == D_quad * N_poly, name="Def_Freq_Term")

# Deflection term: def_term == D^3 * N
model.addConstr(def_term == D_cub * N, name="Def_Def_Term")

# Objective terms
model.addConstr(obj_term_1 == d_sq * D, name="Def_Obj_Term_1")
model.addConstr(obj_term_2 == obj_term_1 * N, name="Def_Obj_Term_2")

# 6.2 Physical Constraints

# Shear Stress Constraint
# 2400 * (stress_frac + 0.615 * d) <= 80000 * pi * d^3
model.addConstr(2400 * (stress_frac + 0.615 * d) <= tau_allowable * PI * d_cub, name="Shear_Stress")

# Natural Frequency Constraint
# f <= 4 Hz implies G * d^2 <= 128 * pi^4 * rho * D^4 * (N^2+2N)
C_freq = 128 * (PI**4) * rho
model.addConstr(G * d_sq <= C_freq * freq_term, name="Natural_Frequency")

# Minimum Compression Constraint
# delta_max >= 0.5 implies 4800 * D^3 * N >= G * d^4
model.addConstr(4800 * def_term >= G * d_quad, name="Min_Compression")

# Winding Ratio (Spring Index) C = D/d
# 4 <= D/d <= 12  =>  4d <= D <= 12d
model.addConstr(D >= 4 * d, name="C_Lower_Bound")
model.addConstr(D <= 12 * d, name="C_Upper_Bound")

# Outer Diameter Constraint
# D + d <= 1.5
model.addConstr(D + d <= outer_diameter_max, name="Outer_Diameter_Max")

# 7. Solve and Print
model.optimize()

if model.Status == GRB.OPTIMAL:
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("FinalAnswer=【No Solution】")