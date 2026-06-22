import gurobipy as gp
from gurobipy import GRB

# ==================== Parameters ====================
alpha = 0.3
beta = 0.8
p_x = 4
p_y = 2
M = 100
N_threshold = 30
tip_rate = 0.1

# ==================== Model ====================
model = gp.Model("CobbDouglas_Retail")

# ==================== Decision Variables ====================
x = model.addVar(vtype=GRB.INTEGER, lb=0, ub=25, name="x")
y = model.addVar(vtype=GRB.INTEGER, lb=0, ub=50, name="y")
z = model.addVar(vtype=GRB.BINARY, name="z")
u = model.addVar(lb=0, ub=GRB.INFINITY, name="u")

# ==================== Auxiliary Variables for Non-linear Objective ====================
# w1 = x^alpha, w2 = y^beta, w3 = w1 * w2 = x^alpha * y^beta
w1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="w1")
w2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="w2")
w3 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="w3")

model.update()
model.Params.NonConvex = 2

# ==================== Objective ====================
model.setObjective(w3, sense=GRB.MAXIMIZE)

# ==================== Constraints ====================
# 1. Budget without fee (always active)
model.addConstr(p_x * x + p_y * y <= M, name="Budget_NoFee")

# 2. Service-fee trigger using indicator constraints
model.addGenConstrIndicator(z, True, x + y >= 31, name="ServiceFee_Trigger_LB")
model.addGenConstrIndicator(z, False, x + y <= 30, name="ServiceFee_Trigger_UB")

# 3. Auxiliary variable u linearization: u = z * (p_x * x + p_y * y)
model.addConstr(u <= p_x * x + p_y * y, name="Aux_u_Upper_xy")
model.addConstr(u <= M * z, name="Aux_u_Upper_Mz")
model.addConstr(u >= p_x * x + p_y * y - M * (1 - z), name="Aux_u_Lower")

# 4. Budget with possible fee
model.addConstr(p_x * x + p_y * y + tip_rate * u <= M, name="Budget_WithFee")

# 5. Power constraints for objective
model.addGenConstrPow(x, w1, alpha, "pow_w1", "")
model.addGenConstrPow(y, w2, beta, "pow_w2", "")

# 6. Multiplication constraint: w3 = w1 * w2
model.addConstr(w1 * w2 == w3, name="utility_product")

# ==================== Solve ====================
model.optimize()

# ==================== Results ====================
if model.status == GRB.OPTIMAL:
    x_val = round(x.X)
    y_val = round(y.X)
    z_val = z.X
    u_val = u.X
    w3_val = w3.X
    print("Optimal Solution Found:")
    print(f"x = {x_val} units")
    print(f"y = {y_val} units")
    print(f"z = {z_val}")
    print(f"u = {u_val:.2f}")
    print(f"Maximum Utility U = {w3_val:.6f}")
    print(f"Total quantity (x+y) = {x_val + y_val}")
    print(f"Total expenditure (including possible fee) = {p_x * x_val + p_y * y_val + tip_rate * u_val:.2f}")
    print(f"FinalAnswer=【{w3_val:.6f}】")
else:
    print("No optimal solution found.")
    print(f"FinalAnswer=【0】")