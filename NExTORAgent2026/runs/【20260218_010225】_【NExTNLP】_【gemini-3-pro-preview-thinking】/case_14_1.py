import gurobipy as gp
from gurobipy import GRB

# 1. Initialize Model
model = gp.Model("Minghui_Production_Optimization")

# 2. Parameters
alpha = 0.5
beta = 0.7
mu = 1
A_t = 20
total_budget = 1000
w = 50
r = 100

# 3. Decision Variables
# L: quantity of labor employed
L = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="L")
# K: quantity of capital employed
K = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="K")

# 4. Auxiliary Substitution Variables
# L_pow represents L^0.5
# K_pow represents K^0.7
# According to coding advice, set bounds to -infinity to +infinity
L_pow = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="L_pow")
K_pow = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="K_pow")

# Enable NonConvex optimization to handle the quadratic objective term (L_pow * K_pow)
model.Params.NonConvex = 2

# 5. Objective Function
# Maximize Q(L,K) = 20 * L^0.5 * K^0.7 * 1
# Replaced with auxiliary variables: 20 * L_pow * K_pow
model.setObjective(A_t * mu * L_pow * K_pow, GRB.MAXIMIZE)

# 6. Constraints

# Budget constraint: 50 L + 100 K <= 1000
model.addConstr(w * L + r * K <= total_budget, name="Budget_Constraint")

# Power constraints (General Constraints)
# Define L_pow = L^alpha
model.addGenConstrPow(L, L_pow, alpha, name="GenConstr_L_pow")

# Define K_pow = K^beta
model.addGenConstrPow(K, K_pow, beta, name="GenConstr_K_pow")

# 7. Solve the model
model.optimize()

# 8. Print Results
if model.status == GRB.OPTIMAL:
    print(f"Optimal Labor (L): {L.X}")
    print(f"Optimal Capital (K): {K.X}")
    print(f"Maximized Output: {model.ObjVal}")
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimization was not successful.")