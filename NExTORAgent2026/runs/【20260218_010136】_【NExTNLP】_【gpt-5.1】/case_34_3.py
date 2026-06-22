import gurobipy as gp
from gurobipy import GRB

# ==========================
# 1. Parameters (from Parameters List)
# ==========================
num_design_variables = 7
design_variables = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
units = 'inches'

model_coefficients = [0.7854, 3.3333, 14.9334, -43.0934, -1.508, 7.4777, 0.7854]
c0, c1, c2, c3, c4, c5, c6 = model_coefficients

meshing_strength_coefficient = 27.0
meshing_strength_threshold = 1.0
shaft_strength_coefficient = 397.5
shaft_strength_threshold = 1.0
pitch_circle_diameter_max = 40.0

Table_1_sizeRestrictions = {
    'Tooth width': [2.5, 3.5],
    'Module': [0.6, 0.8],
    'Number of teeth (integer)': [17, 28],
    'Shaft 1 length': [7.0, 9.0],
    'Shaft 2 length': [7.5, 9.0],
    'Shaft 1 diameter': [2.5, 3.5],
    'Shaft 2 diameter': [5.0, 6.0]
}

# ==========================
# 2. Create model
# ==========================
model = gp.Model("Gearbox_Weight_Optimization")

# Allow nonconvex quadratic / bilinear constraints
model.Params.NonConvex = 2

# ==========================
# 3. Decision variables
# ==========================
x1 = model.addVar(lb=Table_1_sizeRestrictions['Tooth width'][0],
                  ub=Table_1_sizeRestrictions['Tooth width'][1],
                  vtype=GRB.CONTINUOUS, name="x1")
x2 = model.addVar(lb=Table_1_sizeRestrictions['Module'][0],
                  ub=Table_1_sizeRestrictions['Module'][1],
                  vtype=GRB.CONTINUOUS, name="x2")
x3 = model.addVar(lb=Table_1_sizeRestrictions['Number of teeth (integer)'][0],
                  ub=Table_1_sizeRestrictions['Number of teeth (integer)'][1],
                  vtype=GRB.INTEGER, name="x3")
x4 = model.addVar(lb=Table_1_sizeRestrictions['Shaft 1 length'][0],
                  ub=Table_1_sizeRestrictions['Shaft 1 length'][1],
                  vtype=GRB.CONTINUOUS, name="x4")
x5 = model.addVar(lb=Table_1_sizeRestrictions['Shaft 2 length'][0],
                  ub=Table_1_sizeRestrictions['Shaft 2 length'][1],
                  vtype=GRB.CONTINUOUS, name="x5")
x6 = model.addVar(lb=Table_1_sizeRestrictions['Shaft 1 diameter'][0],
                  ub=Table_1_sizeRestrictions['Shaft 1 diameter'][1],
                  vtype=GRB.CONTINUOUS, name="x6")
x7 = model.addVar(lb=Table_1_sizeRestrictions['Shaft 2 diameter'][0],
                  ub=Table_1_sizeRestrictions['Shaft 2 diameter'][1],
                  vtype=GRB.CONTINUOUS, name="x7")

# ==========================
# 4. Auxiliary variables (lb=-inf, ub=+inf)
# ==========================
z2   = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z2")    # x2^2
z3   = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z3")    # x3^2
z6_2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z6_2")  # x6^2
z7_2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z7_2")  # x7^2
z6_3 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z6_3")  # x6^3
z7_3 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z7_3")  # x7^3

p1     = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="p1")      # x1 * x2^2 * x3
p2     = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="p2")      # x1 * x2^2 * x3^2
aux_p1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="aux_p1")  # x1 * z2
aux_p2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="aux_p2")  # x1 * z2
d      = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="d")       # x2 * x3

t6  = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="t6")   # x1 * x6^2
t7  = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="t7")   # x1 * x7^2
s46 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="s46")  # x4 * x6^2
s57 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="s57")  # x5 * x7^2

# ==========================
# 5. Power constraints (GenConstrPow)
# ==========================
model.addGenConstrPow(x2, z2,   2.0, name="gc_z2")
model.addGenConstrPow(x3, z3,   2.0, name="gc_z3")
model.addGenConstrPow(x6, z6_2, 2.0, name="gc_z6_2")
model.addGenConstrPow(x7, z7_2, 2.0, name="gc_z7_2")
model.addGenConstrPow(x6, z6_3, 3.0, name="gc_z6_3")
model.addGenConstrPow(x7, z7_3, 3.0, name="gc_z7_3")

# ==========================
# 6. Bilinear product constraints
# ==========================
model.addConstr(aux_p1 == x1 * z2, name="c_aux_p1")
model.addConstr(p1     == aux_p1 * x3, name="c_p1")

model.addConstr(aux_p2 == x1 * z2, name="c_aux_p2")
model.addConstr(p2     == aux_p2 * z3, name="c_p2")

model.addConstr(d == x2 * x3, name="c_d")

model.addConstr(t6 == x1 * z6_2, name="c_t6")
model.addConstr(t7 == x1 * z7_2, name="c_t7")

model.addConstr(s46 == x4 * z6_2, name="c_s46")
model.addConstr(s57 == x5 * z7_2, name="c_s57")

# ==========================
# 7. Original constraints
# ==========================
model.addConstr(p1 >= meshing_strength_coefficient, name="gear_meshing_strength_limit")
model.addConstr(p2 >= shaft_strength_coefficient,   name="shaft_strength_limit")
model.addConstr(d  <= pitch_circle_diameter_max,    name="pitch_circle_diameter")
model.addConstr(x1 >= 2.0 * x2,                     name="tooth_width_vs_module")

# ==========================
# 8. Objective function (minimize weight)
# ==========================
expr_inner = c1 * z3 + c2 * x3 + c3
objective_expr = (
    c0 * x1 * z2 * expr_inner +
    c4 * (t6 + t7) +
    c5 * (z6_3 + z7_3) +
    c6 * (s46 + s57)
)
model.setObjective(objective_expr, GRB.MINIMIZE)

# ==========================
# 9. Optimize
# ==========================
model.optimize()

# ==========================
# 10. Print results
# ==========================
if model.Status == GRB.OPTIMAL or (model.Status == GRB.INTERRUPTED and model.SolCount > 0):
    x1_val = x1.X
    x2_val = x2.X
    x3_val = x3.X
    x4_val = x4.X
    x5_val = x5.X
    x6_val = x6.X
    x7_val = x7.X
    f_val  = model.ObjVal

    print("Optimal solution found:")
    print(f"x1 (tooth width)           = {x1_val:.6f} {units}")
    print(f"x2 (module)                = {x2_val:.6f} {units}")
    print(f"x3 (number of teeth)       = {x3_val:.6f}")
    print(f"x4 (shaft 1 length)        = {x4_val:.6f} {units}")
    print(f"x5 (shaft 2 length)        = {x5_val:.6f} {units}")
    print(f"x6 (shaft 1 diameter)      = {x6_val:.6f} {units}")
    print(f"x7 (shaft 2 diameter)      = {x7_val:.6f} {units}")
    print(f"Minimum weight objective f = {f_val:.6f}")
else:
    print("No optimal solution found.")
    f_val = float('nan')

# ==========================
# 11. Final answer output
# ==========================
print(f"FinalAnswer=【{f_val}】")