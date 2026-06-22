import gurobipy as gp
from gurobipy import GRB

# Initialize the model
model = gp.Model("Chemical_Production_Optimization")

# 1. Define Parameters
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

# Set NonConvex parameter to handle quadratic equality constraints (auxiliary variables)
model.Params.NonConvex = 2

# 2. Create Decision Variables
# Temperature T
T = model.addVar(lb=T_min, ub=T_max, vtype=GRB.CONTINUOUS, name="T")
# Concentration C
C = model.addVar(lb=C_min, ub=C_max, vtype=GRB.CONTINUOUS, name="C")

# Dependent Variables (Yield, Difficulty, Risk)
# Bounds can be inferred or left as non-negative/free appropriately
Y = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Y")
D = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="D")
H = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="H")

# 3. Create Auxiliary Variables for Quadratic Terms
# We need to compute: (T-80)^2, (C-50)^2, (T-75)^2, (T-50)^2

# Aux vars for linear differences (lb must be -infinity to allow negative differences)
T_diff_80 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="T_diff_80")
C_diff_50 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="C_diff_50")
T_diff_75 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="T_diff_75")
T_diff_50 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="T_diff_50")

# Aux vars for squared terms
T_sq_80 = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="T_sq_80")
C_sq_50 = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="C_sq_50")
T_sq_75 = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="T_sq_75")
T_sq_50 = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="T_sq_50")

# 4. Add Constraints

# Link linear difference variables
model.addConstr(T_diff_80 == T - T_opt, name="Link_T_diff_80")
model.addConstr(C_diff_50 == C - C_opt, name="Link_C_diff_50")
model.addConstr(T_diff_75 == T - difficulty_mean, name="Link_T_diff_75")
# Note: For risk, the problem states (T-50)^2. Since T_min=50, we use that.
model.addConstr(T_diff_50 == T - T_min, name="Link_T_diff_50")

# Link squared variables using General Constraints
model.addGenConstrPow(T_diff_80, T_sq_80, 2, "Pow_T_80")
model.addGenConstrPow(C_diff_50, C_sq_50, 2, "Pow_C_50")
model.addGenConstrPow(T_diff_75, T_sq_75, 2, "Pow_T_75")
model.addGenConstrPow(T_diff_50, T_sq_50, 2, "Pow_T_50")

# Define Functional Relationships
# Yield: Y = 100 - 0.01*(T-80)^2 - 0.02*(C-50)^2
model.addConstr(Y == Y_max - a_T * T_sq_80 - a_C * C_sq_50, name="Yield_Def")

# Difficulty: D = 0.05*(T-75)^2
model.addConstr(D == difficulty_coefficient * T_sq_75, name="Diff_Def")

# Risk: H = 0.001*(T-50)^2 + 0.0431*C
model.addConstr(H == risk_coeff_T * T_sq_50 + risk_coeff_C * C, name="Risk_Def")

# 5. Set Objective Function
# Maximize Z = Y - 0.1*D - 0.5*H
model.setObjective(Y - w_difficulty * D - w_risk * H, GRB.MAXIMIZE)

# 6. Solve the model
model.optimize()

# 7. Print Results
if model.status == GRB.OPTIMAL:
    print(f"Optimal Objective Value: {model.objVal}")
    print(f"Optimal Temperature (T): {T.x}")
    print(f"Optimal Concentration (C): {C.x}")
    print(f"Resulting Yield (Y): {Y.x}")
    print(f"Resulting Difficulty (D): {D.x}")
    print(f"Resulting Risk (H): {H.x}")
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("Optimization was unsuccessful.")