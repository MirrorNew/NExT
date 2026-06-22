import gurobipy as gp

# Create model
model = gp.Model("RadiotherapyDoseOptimization")

# Parameters from list
x1_upper = 60.0
tumor_dose_min = 60.0
healthy_dose_ub = 30.0
tumor_coeff_x1 = 0.8
tumor_coeff_x2 = 1.0
healthy_coeff_x1 = 0.3
healthy_coeff_x2 = 0.6
exponent_healthy = 2.05

# Decision variables
x1 = model.addVar(lb=0, ub=x1_upper, name="x1")
x2 = model.addVar(lb=0, name="x2")
D_T = model.addVar(lb=0, name="D_T")
D_O = model.addVar(lb=0, name="D_O")

# Auxiliary variables for nonlinear terms
Y1 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="Y1")  # (D_T - 60)^2
Y2 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="Y2")  # (D_O)^2.05

# Enable non-convex optimization
model.Params.NonConvex = 2

# Dose definition constraints
model.addConstr(D_T == tumor_coeff_x1 * x1 + tumor_coeff_x2 * x2, name="tumor_dose_def")
model.addConstr(D_O == healthy_coeff_x1 * x1 + healthy_coeff_x2 * x2, name="healthy_dose_def")

# Minimum tumor dose
model.addConstr(D_T >= tumor_dose_min, name="min_tumor_dose")

# Healthy tissue dose upper bound (30 Gy hard constraint)
model.addConstr(D_O <= healthy_dose_ub, name="max_healthy_dose")

# General constraints for nonlinear terms
model.addGenConstrPow(D_T - tumor_dose_min, Y1, 2, name="sq_term")
model.addGenConstrPow(D_O, Y2, exponent_healthy, name="pow_term")

# Objective: minimize f = (D_T - 60)^2 + (D_O)^2.05
model.setObjective(Y1 + Y2, gp.GRB.MINIMIZE)

# Solve
model.optimize()

# Print results
if model.status == gp.GRB.OPTIMAL:
    print("Optimal solution found:")
    print(f"x1 (beam 1 intensity) = {x1.X:.4f} Gy")
    print(f"x2 (beam 2 intensity) = {x2.X:.4f} Gy")
    print(f"D_T (tumor dose) = {D_T.X:.4f} Gy")
    print(f"D_O (healthy tissue dose) = {D_O.X:.4f} Gy")
    print(f"Objective value f = {model.ObjVal:.6f}")
    
    # Final answer is the objective function value
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print(f"Optimization failed with status {model.status}")
    print(f"FinalAnswer=【None】")