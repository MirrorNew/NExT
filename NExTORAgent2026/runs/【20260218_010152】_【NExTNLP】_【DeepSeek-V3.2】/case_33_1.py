import gurobipy as gp

# Parameters from the provided list
T_opt = 80
C_opt = 50
Y_max = 100.0
a_T = 0.01
a_C = 0.02
T_min = 50
T_max = 100
C_min = 20
C_max = 80
T_parabolic_threshold = 100
difficulty_max = 100
difficulty_mean = 75
difficulty_coefficient = 0.05
C_risk_threshold = 60
T_risk_threshold = 85
risk_coeff_T = 0.001
risk_coeff_C = 0.0431
w_difficulty = 0.1
w_risk = 0.5

# Create model
model = gp.Model("ChemicalReactionOptimization")

# Create decision variables
T = model.addVar(lb=T_min, ub=T_max, name="T")
C = model.addVar(lb=C_min, ub=C_max, name="C")

# Create continuous variables for Y, D, H
Y = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="Y")
D = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="D")
H = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="H")

# Create auxiliary variables for squared terms
Y1 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="Y1")  # (T-80)^2
Y2 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="Y2")  # (C-50)^2
Y3 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="Y3")  # (T-75)^2
Y4 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="Y4")  # (T-50)^2

# Add auxiliary expressions for differences
T_minus_80 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="T_minus_80")
T_minus_75 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="T_minus_75")
T_minus_50 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="T_minus_50")
C_minus_50 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="C_minus_50")

model.addConstr(T_minus_80 == T - T_opt, name="T_minus_80_def")
model.addConstr(T_minus_75 == T - difficulty_mean, name="T_minus_75_def")
model.addConstr(T_minus_50 == T - T_min, name="T_minus_50_def")
model.addConstr(C_minus_50 == C - C_opt, name="C_minus_50_def")

# Enable non-convex optimization
model.Params.NonConvex = 2

# Add power constraints for squared terms
model.addGenConstrPow(T_minus_80, Y1, 2, name="pow_T_minus_80")
model.addGenConstrPow(C_minus_50, Y2, 2, name="pow_C_minus_50")
model.addGenConstrPow(T_minus_75, Y3, 2, name="pow_T_minus_75")
model.addGenConstrPow(T_minus_50, Y4, 2, name="pow_T_minus_50")

# Yield definition: Y = 100 - 0.01*(T-80)^2 - 0.02*(C-50)^2
model.addConstr(Y == Y_max - a_T * Y1 - a_C * Y2, name="yield_def")

# Difficulty definition: D = 0.05*(T-75)^2
model.addConstr(D == difficulty_coefficient * Y3, name="difficulty_def")

# Risk factor definition: H = 0.001*(T-50)^2 + 0.0431*C
model.addConstr(H == risk_coeff_T * Y4 + risk_coeff_C * C, name="risk_def")

# Set objective: maximize Z = Y - 0.1*D - 0.5*H
Z = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="Z")
model.addConstr(Z == Y - w_difficulty * D - w_risk * H, name="Z_def")
model.setObjective(Z, gp.GRB.MAXIMIZE)

# Solve the model
model.optimize()

# Print results
if model.status == gp.GRB.OPTIMAL:
    print(f"Optimal temperature T = {T.X:.2f} °C")
    print(f"Optimal concentration C = {C.X:.2f} %")
    print(f"Optimal yield Y = {Y.X:.2f} kg/batch")
    print(f"Operation difficulty D = {D.X:.2f}")
    print(f"Risk factor H = {H.X:.2f}")
    print(f"Objective value Z = {model.ObjVal:.2f}")
    # Format the answer as required
    print(f"FinalAnswer=【T={T.X:.2f}, C={C.X:.2f}, Y={Y.X:.2f}, D={D.X:.2f}, H={H.X:.2f}, Z={model.ObjVal:.2f}】")
else:
    print(f"Model status: {model.status}")
    print("FinalAnswer=【No optimal solution found】")