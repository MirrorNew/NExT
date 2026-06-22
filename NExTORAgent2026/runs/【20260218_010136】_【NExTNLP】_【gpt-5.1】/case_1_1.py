import gurobipy as gp
from gurobipy import GRB

# ============================
# 1. Define parameters
# ============================
b = 200          # production limit per month
c = 100          # storage limit
y_0 = 0          # initial storage
y_3_param = 0    # ending inventory requirement

d_1 = 150
d_2 = 200
d_3 = 212

f_1_coeff = 100
f_1_power = 0.9

f_2_coeff = 100
f_2_power = 0.8

f_3_coeff = 150
f_3_power = 0.5

g_coeff = 20

# ============================
# 2. Create model
# ============================
model = gp.Model("Huaxin_TV_Production_Inventory")

# Allow nonconvex power constraints with fractional exponents
model.Params.NonConvex = 2

# ============================
# 3. Decision variables
# ============================
# Production quantities x1, x2, x3
x = {}
for i in range(1, 4):
    x[i] = model.addVar(lb=0, ub=b, vtype=GRB.CONTINUOUS, name=f"x_{i}")

# Inventory levels y1, y2, y3
y = {}
for i in range(1, 4):
    y[i] = model.addVar(lb=0, ub=c, vtype=GRB.CONTINUOUS, name=f"y_{i}")

# ============================
# 4. Auxiliary variables
# ============================
# Auxiliary variables for power terms: aux_i = x_i^{power}
aux = {}
aux[1] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                      vtype=GRB.CONTINUOUS, name="aux_1")
aux[2] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                      vtype=GRB.CONTINUOUS, name="aux_2")
aux[3] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                      vtype=GRB.CONTINUOUS, name="aux_3")

# Cost variables for production cost terms: c_i = f_i_coeff * aux_i
c_prod = {}
c_prod[1] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                         vtype=GRB.CONTINUOUS, name="c_prod_1")
c_prod[2] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                         vtype=GRB.CONTINUOUS, name="c_prod_2")
c_prod[3] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                         vtype=GRB.CONTINUOUS, name="c_prod_3")

# ============================
# 5. Objective function
# ============================
# Total production cost + holding cost
total_cost = (
    c_prod[1] + c_prod[2] + c_prod[3]
    + g_coeff * (y[1] + y[2] + y[3])
)
model.setObjective(total_cost, GRB.MINIMIZE)

# ============================
# 6. Constraints
# ============================

# 6.1 Power constraints and linking to cost variables
# aux_1 = x_1^0.9, c_prod_1 = 100 * aux_1
model.addGenConstrPow(x[1], aux[1], f_1_power, name="pow_1")
model.addConstr(c_prod[1] == f_1_coeff * aux[1], name="link_cost_1")

# aux_2 = x_2^0.8, c_prod_2 = 100 * aux_2
model.addGenConstrPow(x[2], aux[2], f_2_power, name="pow_2")
model.addConstr(c_prod[2] == f_2_coeff * aux[2], name="link_cost_2")

# aux_3 = x_3^0.5, c_prod_3 = 150 * aux_3
model.addGenConstrPow(x[3], aux[3], f_3_power, name="pow_3")
model.addConstr(c_prod[3] == f_3_coeff * aux[3], name="link_cost_3")

# 6.2 Inventory balance constraints
model.addConstr(y[1] == y_0 + x[1] - d_1, name="inv_balance_1")
model.addConstr(y[2] == y[1] + x[2] - d_2, name="inv_balance_2")
model.addConstr(y[3] == y[2] + x[3] - d_3, name="inv_balance_3")

# 6.3 Ending inventory constraint y_3 = 0
model.addConstr(y[3] == y_3_param, name="ending_inventory")

# Capacity bounds are already handled in variable definitions (0 ≤ x_i ≤ b, 0 ≤ y_i ≤ c)

# ============================
# 7. Solve the model
# ============================
model.optimize()

# ============================
# 8. Print results
# ============================
if model.status == GRB.OPTIMAL:
    x1_opt = x[1].X
    x2_opt = x[2].X
    x3_opt = x[3].X
    y1_opt = y[1].X
    y2_opt = y[2].X
    y3_opt = y[3].X
    obj_val = model.ObjVal

    print("Optimal solution found:")
    print(f"  x1 (production month 1) = {x1_opt:.6f}")
    print(f"  x2 (production month 2) = {x2_opt:.6f}")
    print(f"  x3 (production month 3) = {x3_opt:.6f}")
    print(f"  y1 (inventory end month 1) = {y1_opt:.6f}")
    print(f"  y2 (inventory end month 2) = {y2_opt:.6f}")
    print(f"  y3 (inventory end month 3) = {y3_opt:.6f}")
    print(f"  Minimum total cost Z = {obj_val:.6f}")
else:
    print(f"Optimization ended with status {model.status}")
    x1_opt = float('nan')
    x2_opt = float('nan')
    x3_opt = float('nan')
    obj_val = float('nan')

# ============================
# 9. FinalAnswer output
# ============================
# The question asks for the production plan (monthly production volumes).
# We output them as a tuple (x1, x2, x3).
print(f"FinalAnswer=【({x1_opt:.6f}, {x2_opt:.6f}, {x3_opt:.6f})】")