import gurobipy as gp
from gurobipy import GRB

# Define the target transmission ratio from the parameters list
R_exp = 0.144279

# Define gear tooth ranges from the parameters list
tooth_ranges = {
    'x1': [12, 50],
    'x2': [20, 40],
    'x3': [10, 50],
    'x4': [30, 60]
}

# Create the Gurobi model
model = gp.Model("GearRatioOptimization")

# Enable non-convex solver for bilinear and quadratic general constraints
model.Params.NonConvex = 2

# Create decision variables for tooth numbers
x1 = model.addVar(lb=tooth_ranges['x1'][0], ub=tooth_ranges['x1'][1], vtype=GRB.INTEGER, name="x1")
x2 = model.addVar(lb=tooth_ranges['x2'][0], ub=tooth_ranges['x2'][1], vtype=GRB.INTEGER, name="x2")
x3 = model.addVar(lb=tooth_ranges['x3'][0], ub=tooth_ranges['x3'][1], vtype=GRB.INTEGER, name="x3")
x4 = model.addVar(lb=tooth_ranges['x4'][0], ub=tooth_ranges['x4'][1], vtype=GRB.INTEGER, name="x4")

# Create auxiliary substitution variables as per the coding advice
# Using continuous variables with infinite bounds as requested
y14 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y14")
y23 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y23")
R = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="R")
diff = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="diff")
f = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="f")

# Constraint: y14 = x1 * x4
model.addConstr(y14 == x1 * x4, name="y14_definition")

# Constraint: y23 = x2 * x3
model.addConstr(y23 == x2 * x3, name="y23_definition")

# Constraint: R = y23 / y14, linearized as R * y14 = y23
model.addConstr(R * y14 == y23, name="R_definition")

# Constraint: diff = R - R_exp
model.addConstr(diff == R - R_exp, name="diff_definition")

# Constraint: f = diff^2
model.addGenConstrPow(diff, f, 2, name="f_definition")

# Objective: Minimize the squared deviation f
model.setObjective(f, GRB.MINIMIZE)

# Solve the model
model.optimize()

# Output the result for x1
if model.status == GRB.OPTIMAL:
    x1_val = x1.X
    # Gear teeth are integers, ensuring clean output format
    print(f"FinalAnswer=【{int(round(x1_val))}】")