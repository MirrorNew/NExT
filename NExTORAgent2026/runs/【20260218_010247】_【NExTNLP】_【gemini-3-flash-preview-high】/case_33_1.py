import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
T_opt = 80
C_opt = 50
Y_max = 100.0
a_T = 0.01
a_C = 0.02
T_min = 50
T_max = 100
C_min = 20
C_max = 80
difficulty_mean = 75
difficulty_coefficient = 0.05
risk_coeff_T = 0.001
risk_coeff_C = 0.0431
w_difficulty = 0.1
w_risk = 0.5

# 2. Create decision variables.
model = gp.Model("AntibioticSynthesisOptimization")
model.Params.NonConvex = 2

T = model.addVar(lb=T_min, ub=T_max, vtype=GRB.CONTINUOUS, name="T")
C = model.addVar(lb=C_min, ub=C_max, vtype=GRB.CONTINUOUS, name="C")
Y = model.addVar(lb=0, ub=Y_max, vtype=GRB.CONTINUOUS, name="Y")
D = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="D")
H = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="H")

# 4. Create auxiliary substitution variables.
# The values of these auxiliary variables range from negative infinity to positive infinity.
T_diff_80 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="T_diff_80")
T_diff_80_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="T_diff_80_sq")
C_diff_50 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="C_diff_50")
C_diff_50_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="C_diff_50_sq")
T_diff_75 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="T_diff_75")
T_diff_75_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="T_diff_75_sq")
T_diff_50 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="T_diff_50")
T_diff_50_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="T_diff_50_sq")

# 5. Set up the objective function: maximize Z = Y - 0.1*D - 0.5*H.
model.setObjective(Y - w_difficulty * D - w_risk * H, GRB.MAXIMIZE)

# 6. Add all constraints (including general constraints for quadratic terms).
# Linear constraints for differences
model.addConstr(T_diff_80 == T - T_opt, name="T_diff_80_def")
model.addConstr(C_diff_50 == C - C_opt, name="C_diff_50_def")
model.addConstr(T_diff_75 == T - difficulty_mean, name="T_diff_75_def")
model.addConstr(T_diff_50 == T - T_min, name="T_diff_50_def")

# General constraints for squaring the differences
model.addGenConstrPow(T_diff_80, T_diff_80_sq, 2, name="T_diff_80_sq_gen")
model.addGenConstrPow(C_diff_50, C_diff_50_sq, 2, name="C_diff_50_sq_gen")
model.addGenConstrPow(T_diff_75, T_diff_75_sq, 2, name="T_diff_75_sq_gen")
model.addGenConstrPow(T_diff_50, T_diff_50_sq, 2, name="T_diff_50_sq_gen")

# Definitions of Y, D, and H
model.addConstr(Y == Y_max - a_T * T_diff_80_sq - a_C * C_diff_50_sq, name="yield_definition")
model.addConstr(D == difficulty_coefficient * T_diff_75_sq, name="difficulty_definition")
model.addConstr(H == risk_coeff_T * T_diff_50_sq + risk_coeff_C * C, name="risk_factor_definition")

# 7. Solve the model and print results.
model.optimize()

if model.status == GRB.OPTIMAL:
    # Output the optimized objective value Z as the final answer.
    print(f"FinalAnswer=【{model.objVal}】")