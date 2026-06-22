import gurobipy as gp
from gurobipy import GRB

# 1. Define parameters
R = 10  # radius of the circular square

# 2. Create model
model = gp.Model("MaxInscribedRectangle")

# 3. Create decision variables
L = model.addVar(lb=0, name="L")  # length
W = model.addVar(lb=0, name="W")  # width

# Enable non-convex mode for handling bilinear terms
model.Params.NonConvex = 2

# 4. Set objective: maximize area = L * W
model.setObjective(L * W, sense=GRB.MAXIMIZE)

# 5. Add constraints
# Containment constraint: (L/2)^2 + (W/2)^2 ≤ R^2
model.addConstr((L/2)**2 + (W/2)**2 <= R**2, name="containment")

# 6. Solve the model
model.optimize()

# 7. Print results
if model.status == GRB.OPTIMAL:
    L_val = L.X
    W_val = W.X
    A_val = L_val * W_val
    print("Optimal solution found:")
    print(f"Length L = {L_val:.4f} meters")
    print(f"Width W = {W_val:.4f} meters")
    print(f"Maximum area A = {A_val:.4f} square meters")
    print(f"FinalAnswer=【{A_val:.4f}】")
else:
    print(f"Optimization failed. Status: {model.status}")
    print(f"FinalAnswer=【0】")