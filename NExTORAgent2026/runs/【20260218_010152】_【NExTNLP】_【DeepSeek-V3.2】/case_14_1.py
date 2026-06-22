import gurobipy as gp
from gurobipy import GRB

# Parameters from the provided list
alpha = 0.5
beta = 0.7
mu = 1
A_t = 20
total_budget = 1000
w = 50
r = 100

# Create the model
model = gp.Model("CobbDouglas")

# Create decision variables
L = model.addVar(lb=0, name="L")
K = model.addVar(lb=0, name="K")

# Create auxiliary variables for non-integer powers
# Y1 = L^0.5, Y2 = K^0.7
Y1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Y1")
Y2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Y2")

# Set non-convex parameter
model.Params.NonConvex = 2

# Add constraints for auxiliary variables using general constraints
model.addGenConstrPow(L, Y1, 0.5, name="pow_L")
model.addGenConstrPow(K, Y2, 0.7, name="pow_K")

# Budget constraint
model.addConstr(w * L + r * K <= total_budget, name="budget")

# Set objective: maximize Q = A_t * L^alpha * K^beta * mu
# Since mu = 1, Q = A_t * Y1 * Y2
model.setObjective(A_t * Y1 * Y2, GRB.MAXIMIZE)

# Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print(f"Optimal solution found:")
    print(f"Labor (L) = {L.X:.4f}")
    print(f"Capital (K) = {K.X:.4f}")
    print(f"Objective value (Q) = {model.ObjVal:.4f}")
    print(f"Budget used: {w * L.X + r * K.X:.2f} ≤ {total_budget}")
    # Output the answer to the question
    print(f"FinalAnswer=【{model.ObjVal:.4f}】")
else:
    print(f"No optimal solution found. Status: {model.status}")
    print(f"FinalAnswer=【No optimal solution found】")