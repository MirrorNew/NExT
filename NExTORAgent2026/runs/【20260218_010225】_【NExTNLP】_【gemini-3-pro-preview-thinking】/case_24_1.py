import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs
params = {
    'exponent_healthy_tissue_theoretical': 2.0,
    'exponent_healthy_tissue_actual': 2.05,
    'risk_ratio_excess_dose_to_control': 3,
    'healthy_tissue_dose_target_max': 20.0,
    'minimum_tumor_dose': 60.0,
    'rectal_complication_rate_before': 0.15,
    'rectal_complication_rate_after': 0.05,
    'tumor_control_rate': 0.9,
    'healthy_tissue_protection_gain': 0.2,
    'number_of_beams': 2,
    'tumor_dose_coeffs': {'x1': 0.8, 'x2': 1.0},
    'healthy_dose_coeffs': {'x1': 0.3, 'x2': 0.6},
    'x1_intensity_upper_bound': 60.0,
    'healthy_tissue_dose_upper_bound': 30.0
}

# 2. Create Gurobi Model
model = gp.Model("Radiotherapy_Optimization")
model.Params.NonConvex = 2  # Enable handling of non-convex/power constraints

# 3. Create Decision Variables
# x1: Intensity of beam 1 (0 <= x1 <= 60)
x1 = model.addVar(lb=0, ub=params['x1_intensity_upper_bound'], vtype=GRB.CONTINUOUS, name="x1")
# x2: Intensity of beam 2 (0 <= x2)
x2 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x2")
# D_T: Tumor area absorbed dose
D_T = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="D_T")
# D_O: Healthy tissue absorbed dose
D_O = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="D_O")

# 4. Create Auxiliary Variables
# For the term (D_T - 60)
diff_DT = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="diff_DT")
# For the term (D_T - 60)^2
obj_term1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="obj_term1")
# For the term (D_O)^2.05
obj_term2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="obj_term2")

# 5. Set up Constraints

# Tumor Dose Definition: D_T = 0.8*x1 + 1.0*x2
model.addConstr(D_T == params['tumor_dose_coeffs']['x1'] * x1 + params['tumor_dose_coeffs']['x2'] * x2, "TumorDoseDef")

# Healthy Tissue Dose Definition: D_O = 0.3*x1 + 0.6*x2
model.addConstr(D_O == params['healthy_dose_coeffs']['x1'] * x1 + params['healthy_dose_coeffs']['x2'] * x2, "HealthyDoseDef")

# Minimum Tumor Dose: D_T >= 60
model.addConstr(D_T >= params['minimum_tumor_dose'], "MinTumorDose")

# Maximum Healthy Tissue Dose: D_O <= 30
model.addConstr(D_O <= params['healthy_tissue_dose_upper_bound'], "MaxHealthyDose")

# Auxiliary Constraints for Objective Terms
# diff_DT = D_T - 60
model.addConstr(diff_DT == D_T - params['minimum_tumor_dose'], "DiffDT_Def")

# obj_term1 = (diff_DT)^2
model.addGenConstrPow(diff_DT, obj_term1, 2.0, "Pow_DT")

# obj_term2 = (D_O)^2.05
# Note: D_O is non-negative, so we can raise it to a fractional power
model.addGenConstrPow(D_O, obj_term2, params['exponent_healthy_tissue_actual'], "Pow_DO")

# 6. Set up Objective Function
# Minimize f = (D_T - 60)^2 + (D_O)^2.05
model.setObjective(obj_term1 + obj_term2, GRB.MINIMIZE)

# 7. Solve the model and print results
model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"Optimal Objective Value: {model.ObjVal}")
    print(f"x1: {x1.X}, x2: {x2.X}")
    print(f"D_T: {D_T.X}, D_O: {D_O.X}")
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimization was not successful.")