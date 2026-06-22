import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs
# From Parameters List:
# exponent_healthy_tissue_actual = 2.05
# minimum_tumor_dose = 60.0
# healthy_tissue_dose_target_max = 20.0 (Target, not hard constraint)
# healthy_tissue_dose_upper_bound = 30.0 (Hard constraint)
# x1_intensity_upper_bound = 60.0
# tumor_dose_coeffs = {'x1': 0.8, 'x2': 1.0}
# healthy_dose_coeffs = {'x1': 0.3, 'x2': 0.6}

exponent_actual = 2.05
min_tumor_dose = 60.0
healthy_dose_ub = 30.0
x1_ub = 60.0
tumor_coeffs_x1 = 0.8
tumor_coeffs_x2 = 1.0
healthy_coeffs_x1 = 0.3
healthy_coeffs_x2 = 0.6

# 2. Create the model
model = gp.Model("Radiotherapy_Optimization")

# 3. Create decision variables
# x1, x2: Beam intensities
# DT, DO: Absorbed doses
x1 = model.addVar(lb=0, ub=x1_ub, vtype=GRB.CONTINUOUS, name="x1")
x2 = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x2")
DT = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="DT")
DO = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="DO")

# 4. Create auxiliary substitution variables
# DT_diff = DT - 60
# Y1 = (DT - 60)^2
# Y2 = (DO)^2.05
DT_diff = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="DT_diff")
Y1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Y1")
Y2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Y2")

# 5. Set up the objective function
# minimize f = (DT - 60)^2 + DO^2.05
model.setObjective(Y1 + Y2, GRB.MINIMIZE)

# 6. Add all constraints
# Dose definitions
model.addConstr(DT == tumor_coeffs_x1 * x1 + tumor_coeffs_x2 * x2, "Tumor_Dose_Definition")
model.addConstr(DO == healthy_coeffs_x1 * x1 + healthy_coeffs_x2 * x2, "Healthy_Tissue_Dose_Definition")

# Minimum tumor dose and Healthy tissue dose upper limit
model.addConstr(DT >= min_tumor_dose, "Min_Tumor_Dose_Constraint")
model.addConstr(DO <= healthy_dose_ub, "Max_Healthy_Tissue_Dose_Constraint")

# Auxiliary variable definitions
model.addConstr(DT_diff == DT - min_tumor_dose, "DT_Difference_Constraint")

# General constraints for the nonlinear objective terms
model.Params.NonConvex = 2
model.addGenConstrPow(DT_diff, Y1, 2.0, "Y1_Constraint")
model.addGenConstrPow(DO, Y2, exponent_actual, "Y2_Constraint")

# 7. Solve the model and print results
model.optimize()

if model.Status == GRB.OPTIMAL:
    obj_val = model.ObjVal
    print(f"Optimal Objective Value: {obj_val}")
    print(f"FinalAnswer=【{obj_val}】")
else:
    print("Optimal solution not found.")