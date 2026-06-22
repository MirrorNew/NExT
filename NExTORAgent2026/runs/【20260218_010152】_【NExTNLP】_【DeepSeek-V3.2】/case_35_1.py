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

# Create auxiliary substitution variables
Y1 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="Y1")
Y2 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="Y2")
Z = model.addVar(lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="Z")

# Set objective
model.setObjective(f, gp.GRB.MINIMIZE)

# Add constraints
# Constraint: Y1 = x1 * x4
model.addConstr(Y1 == x1 * x4, name="Y1_def")

# Constraint: Y2 = x2 * x3
model.addConstr(Y2 == x2 * x3, name="Y2_def")

# Constraint: R * Y1 = Y2  (equivalent to R = Y2 / Y1 = (x2*x3)/(x1*x4))
model.addConstr(R * Y1 == Y2, name="R_def")

# Constraint: Z = (R_exp - R)^2
# Enable non-convex mode for power constraint
model.Params.NonConvex = 2
diff = R_exp - R
model.addGenConstrPow(diff, Z, 2, name="Z_def")

# Constraint: f = Z
model.addConstr(f == Z, name="f_def")

# Solve the model
model.optimize()

# Print results
if model.status == gp.GRB.OPTIMAL:
    print("Optimal solution found:")
    print(f"x1 (Gear 1 teeth) = {x1.x}")
    print(f"x2 (Gear 2 teeth) = {x2.x}")
    print(f"x3 (Gear 3 teeth) = {x3.x}")
    print(f"x4 (Gear 4 teeth) = {x4.x}")
    print(f"Actual transmission ratio R = {R.x:.6f}")
    print(f"Squared deviation f = {f.x:.10f}")
    print(f"Absolute deviation |R_exp - R| = {abs(R_exp - R.x):.6f}")
    print(f"Percentage deviation = {abs(R_exp - R.x)/R_exp*100:.4f}%")
    
    # Output the answer for x1 as requested
    print(f"FinalAnswer=【{x1.x}】")
else:
    print(f"Model status: {model.status}")
    print("No optimal solution found")
    print(f"FinalAnswer=【None】")