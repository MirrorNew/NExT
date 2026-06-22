import gurobipy as gp
from gurobipy import GRB

# ===========================
# 1. Parameters (given list)
# ===========================
deviation_threshold = 0.005
error_tolerance = 0.001
R_exp = 0.144279
Table_1_Tooth_Range = {
    'x1': [12, 50],
    'x2': [20, 40],
    'x3': [10, 50],
    'x4': [30, 60]
}

x1_min, x1_max = Table_1_Tooth_Range['x1']
x2_min, x2_max = Table_1_Tooth_Range['x2']
x3_min, x3_max = Table_1_Tooth_Range['x3']
x4_min, x4_max = Table_1_Tooth_Range['x4']

# ===========================
# 2. Create model
# ===========================
model = gp.Model("Gear_Ratio_Optimization")

# Allow nonconvex expressions (needed for bilinear/trilinear terms)
model.Params.NonConvex = 2

# ===========================
# 3. Decision variables
# ===========================
x1 = model.addVar(vtype=GRB.INTEGER, lb=x1_min, ub=x1_max, name="x1")
x2 = model.addVar(vtype=GRB.INTEGER, lb=x2_min, ub=x2_max, name="x2")
x3 = model.addVar(vtype=GRB.INTEGER, lb=x3_min, ub=x3_max, name="x3")
x4 = model.addVar(vtype=GRB.INTEGER, lb=x4_min, ub=x4_max, name="x4")

R = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="R")

# ===========================
# 4. Auxiliary substitution variables
# ===========================
# To avoid a trilinear term, introduce bilinear auxiliaries
z14 = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="z14")  # z14 = x1 * x4
z2R = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="z2R")  # z2R = R * z14

# t = R_exp - R (free)
t = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t")

# g = t^2 (squared deviation)
g = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="g")

# ===========================
# 5. Objective function
# ===========================
model.setObjective(g, GRB.MINIMIZE)

# ===========================
# 6. Constraints
# ===========================

# 6.1 Bilinear definitions to replace trilinear term
# z14 = x1 * x4
model.addConstr(z14 == x1 * x4, name="z14_definition")

# z2R = R * z14
model.addConstr(z2R == R * z14, name="z2R_definition")

# 6.2 Ratio definition: R * x1 * x4 = x2 * x3  ->  z2R = x2 * x3
model.addConstr(z2R == x2 * x3, name="Ratio_Definition")

# 6.3 t = R_exp - R
model.addConstr(t == R_exp - R, name="t_definition")

# 6.4 g = t^2
model.addGenConstrPow(t, g, 2.0, name="g_definition")

# ===========================
# 7. Solve model
# ===========================
model.optimize()

# ===========================
# 8. Print results
# ===========================
x1_val = None

if model.Status == GRB.OPTIMAL:
    x1_val = int(round(x1.X))
    x2_val = int(round(x2.X))
    x3_val = int(round(x3.X))
    x4_val = int(round(x4.X))
    R_val = R.X
    g_val = g.X

    print("Optimal solution found:")
    print(f"x1 (Gear 1 teeth) = {x1_val}")
    print(f"x2 (Gear 2 teeth) = {x2_val}")
    print(f"x3 (Gear 3 teeth) = {x3_val}")
    print(f"x4 (Gear 4 teeth) = {x4_val}")
    print(f"Actual ratio R     = {R_val:.9f}")
    print(f"Expected ratio     = {R_exp:.9f}")
    print(f"Squared deviation  = {g_val:.12e}")
else:
    print(f"Optimization ended with status {model.Status}")

# ===========================
# 9. Final answer for x1
# ===========================
print(f"FinalAnswer=【{x1_val}】")