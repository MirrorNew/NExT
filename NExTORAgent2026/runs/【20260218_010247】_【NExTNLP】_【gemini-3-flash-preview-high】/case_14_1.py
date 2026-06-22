import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and any other necessary packages.

# 2. Define all parameter matrices and data inputs.
# Parameters from the provided list
alpha = 0.5
beta = 0.7
mu = 1
A_t = 20
total_budget = 1000
w = 50
r = 100

# 3. Create decision variables.
model = gp.Model("ProductionOptimization")
L = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="L")
K = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="K")

# 4. Create auxiliary substitution variables.
# As instructed, auxiliary variables should range from negative infinity to positive infinity.
P_L = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="P_L") # L^0.5
P_K = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="P_K") # K^0.7
Z_prod = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Z_prod") # P_L * P_K

# Gurobi parameter for handling non-convex constraints (bilinear and power)
model.Params.NonConvex = 2

# 5. Set up the objective function.
# Q(L,K) = A_t * mu * L^alpha * K^beta = 20 * 1 * P_L * P_K
model.setObjective(A_t * mu * Z_prod, GRB.MAXIMIZE)

# 6. Add all constraints (including gen-constr and indicator constraints).
# Budget constraint: 50L + 100K <= 1000
model.addConstr(w * L + r * K <= total_budget, name="BudgetConstraint")

# Power relationship: P_L = L^alpha (L^0.5)
# Using addGenConstrPow(X, Y, exponent) -> Y = X^exponent
model.addGenConstrPow(L, P_L, alpha, "powL")

# Power relationship: P_K = K^beta (K^0.7)
# Using addGenConstrPow(X, Y, exponent) -> Y = X^exponent
model.addGenConstrPow(K, P_K, beta, "powK")

# Product relationship: Z_prod = P_L * P_K
# Using model.addConstr for bilinear terms (supported by NonConvex=2)
model.addConstr(Z_prod == P_L * P_K, name="ProductConstraint")

# 7. Solve the model and print results.
model.optimize()

# Print the final output (Q)
if model.Status == GRB.OPTIMAL:
    output_q = model.ObjVal
    print(f"FinalAnswer=【{output_q}】")
else:
    print("Optimization was not successful.")