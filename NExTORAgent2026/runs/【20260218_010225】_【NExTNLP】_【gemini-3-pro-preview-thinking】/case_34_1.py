import gurobipy as gp
from gurobipy import GRB

# 1. Create model and set NonConvex parameter
model = gp.Model("Gearbox_Optimization")
model.Params.NonConvex = 2  # Handle non-convex quadratic constraints and objectives

# 2. Define Decision Variables (x1 to x7) based on the problem description
# x1: gear tooth width, Range: [2.5, 3.5]
x1 = model.addVar(lb=2.5, ub=3.5, vtype=GRB.CONTINUOUS, name="x1")

# x2: module, Range: [0.6, 0.8]
x2 = model.addVar(lb=0.6, ub=0.8, vtype=GRB.CONTINUOUS, name="x2")

# x3: number of teeth, Range: [17, 28], Integer
x3 = model.addVar(lb=17, ub=28, vtype=GRB.INTEGER, name="x3")

# x4: shaft 1 length, Range: [7, 9]
x4 = model.addVar(lb=7.0, ub=9.0, vtype=GRB.CONTINUOUS, name="x4")

# x5: shaft 2 length, Range: [7.5, 9]
x5 = model.addVar(lb=7.5, ub=9.0, vtype=GRB.CONTINUOUS, name="x5")

# x6: shaft 1 diameter, Range: [2.5, 3.5]
x6 = model.addVar(lb=2.5, ub=3.5, vtype=GRB.CONTINUOUS, name="x6")

# x7: shaft 2 diameter, Range: [5, 6]
x7 = model.addVar(lb=5.0, ub=6.0, vtype=GRB.CONTINUOUS, name="x7")

# 3. Create Auxiliary Variables for Non-Linear Terms (Substitution)
# We set bounds to -inf, +inf as requested, though physically they are positive.

# Powers: x2^2, x3^2, x6^2, x7^2, x6^3, x7^3
y_x2_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y_x2_sq")
y_x3_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y_x3_sq")
y_x6_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y_x6_sq")
y_x7_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y_x7_sq")
y_x6_cub = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y_x6_cub")
y_x7_cub = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y_x7_cub")

# Products needed for Objective and Constraints
# Term: x1 * x2^2
v_x1_x2sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="v_x1_x2sq")
# Term: x1 * x2^2 * x3
v_x1_x2sq_x3 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="v_x1_x2sq_x3")
# Term: x1 * x2^2 * x3^2
v_x1_x2sq_x3sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="v_x1_x2sq_x3sq")
# Term: x1 * x6^2
v_x1_x6sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="v_x1_x6sq")
# Term: x1 * x7^2
v_x1_x7sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="v_x1_x7sq")
# Term: x4 * x6^2
v_x4_x6sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="v_x4_x6sq")
# Term: x5 * x7^2
v_x5_x7sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="v_x5_x7sq")

# 4. Link Auxiliary Variables with General Constraints and Quadratic Constraints
# Power constraints
model.addGenConstrPow(x2, y_x2_sq, 2)
model.addGenConstrPow(x3, y_x3_sq, 2)
model.addGenConstrPow(x6, y_x6_sq, 2)
model.addGenConstrPow(x7, y_x7_sq, 2)
model.addGenConstrPow(x6, y_x6_cub, 3)
model.addGenConstrPow(x7, y_x7_cub, 3)

# Product constraints (Substitution)
# v_x1_x2sq = x1 * y_x2_sq
model.addConstr(v_x1_x2sq == x1 * y_x2_sq)
# v_x1_x2sq_x3 = v_x1_x2sq * x3
model.addConstr(v_x1_x2sq_x3 == v_x1_x2sq * x3)
# v_x1_x2sq_x3sq = v_x1_x2sq * y_x3_sq
model.addConstr(v_x1_x2sq_x3sq == v_x1_x2sq * y_x3_sq)
# v_x1_x6sq = x1 * y_x6_sq
model.addConstr(v_x1_x6sq == x1 * y_x6_sq)
# v_x1_x7sq = x1 * y_x7_sq
model.addConstr(v_x1_x7sq == x1 * y_x7_sq)
# v_x4_x6sq = x4 * y_x6_sq
model.addConstr(v_x4_x6sq == x4 * y_x6_sq)
# v_x5_x7sq = x5 * y_x7_sq
model.addConstr(v_x5_x7sq == x5 * y_x7_sq)

# 5. Set up the Objective Function
# Original: f = 0.7854*x1*x2^2*(3.3333*x3^2 + 14.9334*x3 - 43.0934) - 1.508*x1*(x6^2+x7^2) + 7.4777*(x6^3+x7^3) + 0.7854*(x4*x6^2 + x5*x7^2)
# Expanded terms using auxiliary variables:
# Term 1 expansion: 0.7854 * (3.3333 * x1*x2^2*x3^2 + 14.9334 * x1*x2^2*x3 - 43.0934 * x1*x2^2)
term1 = 0.7854 * (3.3333 * v_x1_x2sq_x3sq + 14.9334 * v_x1_x2sq_x3 - 43.0934 * v_x1_x2sq)
term2 = -1.508 * (v_x1_x6sq + v_x1_x7sq)
term3 = 7.4777 * (y_x6_cub + y_x7_cub)
term4 = 0.7854 * (v_x4_x6sq + v_x5_x7sq)

model.setObjective(term1 + term2 + term3 + term4, GRB.MINIMIZE)

# 6. Add Constraints

# Gear-meshing strength limit: x1 * x2^2 * x3 >= 27
model.addConstr(v_x1_x2sq_x3 >= 27, "GearMeshingStrengthLimit")

# Shaft-strength limit: x1 * x2^2 * x3^2 >= 397.5
model.addConstr(v_x1_x2sq_x3sq >= 397.5, "ShaftStrengthLimit")

# Pitch circle diameter constraint: x2 * x3 <= 40
model.addConstr(x2 * x3 <= 40, "PitchCircleDiameter")

# Tooth width vs module constraint: x1 >= 2 * x2
model.addConstr(x1 >= 2 * x2, "ToothWidthVsModule")

# 7. Solve and print results
model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print(f"Optimization failed with status {model.status}")