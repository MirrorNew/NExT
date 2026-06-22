import gurobipy as gp
from gurobipy import GRB

# ==========================
# 1. Define parameters (from Parameters List ONLY)
# ==========================
exponent_healthy_tissue_theoretical = 2.0
exponent_healthy_tissue_actual = 2.05
risk_ratio_excess_dose_to_control = 3
healthy_tissue_dose_target_max = 20.0
minimum_tumor_dose = 60.0
rectal_complication_rate_before = 0.15
rectal_complication_rate_after = 0.05
tumor_control_rate = 0.9
healthy_tissue_protection_gain = 0.2
number_of_beams = 2

tumor_dose_coeffs = {'x1': 0.8, 'x2': 1.0}
healthy_dose_coeffs = {'x1': 0.3, 'x2': 0.6}
x1_intensity_upper_bound = 60.0
healthy_tissue_dose_upper_bound = 30.0  # additional safety upper bound

# ==========================
# 2. Create model
# ==========================
model = gp.Model("IMRT_two_beam_dose_optimization")

# Model is nonconvex due to non-integer exponent 2.05
model.Params.NonConvex = 2

# ==========================
# 3. Decision variables
# ==========================
# Beam intensities
x1 = model.addVar(lb=0.0, ub=x1_intensity_upper_bound, name="x1")
x2 = model.addVar(lb=0.0, ub=GRB.INFINITY, name="x2")

# Doses
D_T = model.addVar(lb=0.0, ub=GRB.INFINITY, name="D_T")
D_O = model.addVar(lb=0.0, ub=GRB.INFINITY, name="D_O")

# ==========================
# 4. Auxiliary / substitution variables
# ==========================
# v_DT = D_T - minimum_tumor_dose
v_DT = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="v_DT")

# z_DT = (D_T - 60)^2
z_DT = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="z_DT")

# z_DO = (D_O)^(2.05)
z_DO = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="z_DO")

# Objective variable f
f = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="f")

# ==========================
# 5. Objective function: minimize f = (D_T - 60)^2 + (D_O)^2.05
# ==========================
model.setObjective(f, GRB.MINIMIZE)

# ==========================
# 6. Constraints
# ==========================

# 6.1 Dose–intensity relationships
model.addConstr(
    D_T == tumor_dose_coeffs['x1'] * x1 + tumor_dose_coeffs['x2'] * x2,
    name="TumorDoseDef"
)

model.addConstr(
    D_O == healthy_dose_coeffs['x1'] * x1 + healthy_dose_coeffs['x2'] * x2,
    name="HealthyDoseDef"
)

# 6.2 Clinical dose requirements
# Minimum tumor dose: D_T ≥ 60
model.addConstr(D_T >= minimum_tumor_dose, name="MinTumorDose")

# Healthy tissue dose target: D_O ≤ 20
model.addConstr(D_O <= healthy_tissue_dose_target_max, name="HealthyDoseTargetMax")

# Safety upper bound: D_O ≤ 30 (redundant but explicitly included)
model.addConstr(D_O <= healthy_tissue_dose_upper_bound, name="HealthyDoseUpperBound")

# 6.3 Non-negativity (x1/x2 non-negativity already partially handled by lb)
model.addConstr(x1 >= 0.0, name="x1_nonneg")
model.addConstr(x2 >= 0.0, name="x2_nonneg")
model.addConstr(D_T >= 0.0, name="DT_nonneg")
model.addConstr(D_O >= 0.0, name="DO_nonneg")

# 6.4 Auxiliary definitions for nonlinear terms
# v_DT = D_T - minimum_tumor_dose
model.addConstr(v_DT == D_T - minimum_tumor_dose, name="v_DT_def")

# z_DT = (v_DT)^2
model.addGenConstrPow(v_DT, z_DT, 2.0, name="TumorQuadraticTerm")

# z_DO = (D_O)^(2.05)
model.addGenConstrPow(D_O, z_DO, exponent_healthy_tissue_actual, name="HealthyPowerTerm")

# f = z_DT + z_DO
model.addConstr(f == z_DT + z_DO, name="ObjectiveLink")

# ==========================
# 7. Solve model
# ==========================
model.optimize()

# ==========================
# 8. Print results
# ==========================
if model.Status == GRB.OPTIMAL:
    print("Optimal solution found.")
    print(f"x1 (beam 1 intensity) = {x1.X:.6f} Gy")
    print(f"x2 (beam 2 intensity) = {x2.X:.6f} Gy")
    print(f"Tumor dose D_T        = {D_T.X:.6f} Gy")
    print(f"Healthy dose D_O      = {D_O.X:.6f} Gy")
    print(f"(D_T - 60)^2 term     = {z_DT.X:.6f}")
    print(f"(D_O)^2.05 term       = {z_DO.X:.6f}")
    print(f"Objective f           = {f.X:.6f}")

    # FinalAnswer is the minimized value of the dosing specification function f
    print(f"FinalAnswer=【{f.X:.6f}】")
else:
    # If not optimal, still output a FinalAnswer value
    if model.SolCount > 0:
        val = model.ObjVal
    else:
        val = float('nan')
    print(f"No optimal solution. Model status: {model.Status}")
    print(f"FinalAnswer=【{val}】")