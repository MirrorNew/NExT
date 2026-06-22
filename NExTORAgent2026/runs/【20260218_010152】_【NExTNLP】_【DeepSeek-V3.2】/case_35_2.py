import gurobipy as gp

# Define parameters from Parameters List
R_exp = 0.144279  # 1/6.931
tooth_ranges = {'x1': [12, 50], 'x2': [20, 40], 'x3': [10, 50], 'x4': [30, 60]}

# Create model
model = gp.Model("GearTeethOptimization")

# Create decision variables (integer variables for tooth counts)
x1 = model.addVar(lb=tooth_ranges['x1'][0], ub=tooth_ranges['x1'][1], vtype=gp.GRB.INTEGER, name="x1")
x2 = model.addVar(lb=tooth_ranges['x2'][0], ub=tooth_ranges['x2'][1], vtype=gp.GRB.INTEGER, name="x2")
x3 = model.addVar(lb=tooth_ranges['x3'][0], ub=tooth_ranges['x3'][1], vtype=gp.GRB.INTEGER, name="x3")
x4 = model.addVar(lb=tooth_ranges['x4'][0], ub=tooth_ranges['x4'][1], vtype=gp.GRB.INTEGER, name="x4")

# Create continuous variables
R = model.addVar(lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="R")
f = model.addVar(lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="f")

# Set objective
model.setObjective(f, gp.GRB.MINIMIZE)

# Add constraints
# Constraint: R = (x2 * x3) / (x1 * x4)
# Using direct multiplication to avoid division by variable
model.addConstr(R * x1 * x4 == x2 * x3, name="R_def")

# Constraint: f = (R_exp - R)^2
# Enable non-convex mode for power constraint
model.Params.NonConvex = 2
model.addGenConstrPow(R_exp - R, f, 2, name="f_def")

# Solve the model
model.optimize()

# Print results and output answer
if model.status == gp.GRB.OPTIMAL:
    print(f"FinalAnswer=【{x1.x}】")
else:
    print(f"FinalAnswer=【None】")