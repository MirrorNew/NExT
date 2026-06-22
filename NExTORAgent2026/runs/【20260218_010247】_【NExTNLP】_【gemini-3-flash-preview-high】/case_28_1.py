import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
total_vehicles = 2100
capacity_A = 1000
capacity_B = 1200
threshold_A = 900
threshold_B = 1080  # 0.9 * 1200
idle_time_A = 10
idle_time_B = 12
coef_A = 0.0001
coef_B = 0.00008
wait_multiplier = 1.05

# 2. Create the model
model = gp.Model("TrafficAllocation")
model.Params.NonConvex = 2

# 3. Create decision variables.
f_A = model.addVar(lb=0, ub=capacity_A, vtype=GRB.CONTINUOUS, name="f_A")
f_B = model.addVar(lb=0, ub=capacity_B, vtype=GRB.CONTINUOUS, name="f_B")
T_A = model.addVar(lb=10, vtype=GRB.CONTINUOUS, name="T_A")
T_B = model.addVar(lb=12, vtype=GRB.CONTINUOUS, name="T_B")
y_A = model.addVar(vtype=GRB.BINARY, name="y_A")
y_B = model.addVar(vtype=GRB.BINARY, name="y_B")

# 4. Create auxiliary substitution or indicator variables in coding advice.
# (The values of these auxiliary variables should range from negative infinity to positive infinity, lb=-GRB.INFINITY, ub=GRB.INFINITY).
fA2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="fA2")
fB2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="fB2")
fA_TA = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="fA_TA")
fB_TB = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="fB_TB")

# 5. Set up the objective function.
model.setObjective(fA_TA + fB_TB, GRB.MINIMIZE)

# 6. Add all constraints (including gen‐constr and indicator constraints).
# Flow conservation
model.addConstr(f_A + f_B == total_vehicles, "FlowConservation")

# Variable substitution for squares
model.addGenConstrPow(f_A, fA2, 2)
model.addGenConstrPow(f_B, fB2, 2)

# Substitution for bilinear products in objective
model.addConstr(fA_TA == f_A * T_A)
model.addConstr(fB_TB == f_B * T_B)

# Road A threshold logic and travel time
model.addGenConstrIndicator(y_A, 1, f_A >= 900)
model.addGenConstrIndicator(y_A, 0, f_A <= 899.99)
model.addGenConstrIndicator(y_A, 1, T_A == wait_multiplier * (idle_time_A + coef_A * fA2))
model.addGenConstrIndicator(y_A, 0, T_A == idle_time_A + coef_A * fA2)

# Road B threshold logic and travel time
model.addGenConstrIndicator(y_B, 1, f_B >= 1080)
model.addGenConstrIndicator(y_B, 0, f_B <= 1079.99)
model.addGenConstrIndicator(y_B, 1, T_B == wait_multiplier * (idle_time_B + coef_B * fB2))
model.addGenConstrIndicator(y_B, 0, T_B == idle_time_B + coef_B * fB2)

# 7. Solve the model and print results.
model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"FinalAnswer=【{f_A.X}】")
else:
    print("Optimal solution not found.")