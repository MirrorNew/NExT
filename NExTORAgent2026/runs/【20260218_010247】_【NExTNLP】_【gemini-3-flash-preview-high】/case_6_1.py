import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs
R = 10

# 2. Create the model
model = gp.Model("FountainPoolOptimization")
model.Params.NonConvex = 2

# 3. Create decision variables
L = model.addVar(lb=0, name="L")
W = model.addVar(lb=0, name="W")

# 4. Create auxiliary substitution variables
# (As per advice, auxiliary variable bounds are set from -infinity to +infinity)
L_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="L_sq")
W_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="W_sq")
A = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="A")

# 5. Set up the objective function
model.setObjective(A, GRB.MAXIMIZE)

# 6. Add all constraints
# Defining relationships for auxiliary variables using general constraints
model.addGenConstrPow(L, L_sq, 2)  # L_sq = L^2
model.addGenConstrPow(W, W_sq, 2)  # W_sq = W^2

# Area definition: A = L * W
model.addConstr(L * W == A)

# Containment constraint: (L/2)^2 + (W/2)^2 <= R^2  => L^2 + W^2 <= 4 * R^2
# Simplified containment constraint using auxiliary square variables
model.addConstr(L_sq + W_sq <= 4 * (R**2))

# 7. Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print(f"Optimal Length (L): {L.X}")
    print(f"Optimal Width (W): {W.X}")
    print(f"Maximum Area (A): {A.X}")
    print(f"FinalAnswer=【{A.X}】")
else:
    print("Optimization was not successful.")