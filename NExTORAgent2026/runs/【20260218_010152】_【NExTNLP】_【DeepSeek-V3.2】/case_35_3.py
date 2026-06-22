import gurobipy as gp

# Define parameters from Parameters List
R_exp = 0.144279  # 1/6.931
x1_range = [12, 50]
x2_range = [20, 40]
x3_range = [10, 50]
x4_range = [30, 60]

# Create model
model = gp.Model("GearTeethOptimization")

# Create decision variables (integer variables for tooth counts)
x1 = model.addVar(lb=x1_range[0], ub=x1_range[1], vtype=gp.GRB.INTEGER, name="x1")
x2 = model.addVar(lb=x2_range[0], ub=x2_range[1], vtype=gp.GRB.INTEGER, name="x2")
x3 = model.addVar(lb=x3_range[0], ub=x3_range[1], vtype=gp.GRB.INTEGER, name="x3")
x4 = model.addVar(lb=x4_range[0], ub=x4_range[1], vtype=gp.GRB.INTEGER, name="x4")

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
# Enable non-convex mode for power constraint (required for quadratic constraints)
model.Params.NonConvex = 2
diff = R_exp - R
model.addGenConstrPow(diff, f, 2, name="f_def")

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
    print(f"FinalAnswer=【{int(round(x1.x))}】")
else:
    print(f"Model status: {model.status}")
    print("No optimal solution found")
    print(f"FinalAnswer=【None】")