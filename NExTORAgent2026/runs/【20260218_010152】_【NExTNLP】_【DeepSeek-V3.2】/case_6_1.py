import gurobipy as gp
from gurobipy import GRB

# 1. Define parameters
R = 10  # radius of the circular square

# 2. Create model
model = gp.Model("MaxInscribedRectangle")

# 3. Create decision variables
L = model.addVar(lb=0, name="L")  # length
W = model.addVar(lb=0, name="W")  # width

# 4. Create auxiliary variables
A = model.addVar(lb=0, name="A")  # area = L * W
X = model.addVar(lb=0, name="X")  # X = (L/2)^2
Y = model.addVar(lb=0, name="Y")  # Y = (W/2)^2

# Enable non-convex mode for handling bilinear and quadratic terms
model.Params.NonConvex = 2

# 5. Set objective: maximize area
model.setObjective(A, sense=GRB.MAXIMIZE)

# 6. Add constraints
# Area constraint: A = L * W
model.addGenConstrMul(L, W, A, name="area_def")

# X = (L/2)^2
half_L = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="half_L")
model.addConstr(half_L == L / 2, name="half_L_def")
model.addGenConstrPow(half_L, X, 2, name="L_sq_constraint")

# Y = (W/2)^2
half_W = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="half_W")
model.addConstr(half_W == W / 2, name="half_W_def")
model.addGenConstrPow(half_W, Y, 2, name="W_sq_constraint")

# Containment constraint: (L/2)^2 + (W/2)^2 <= R^2
model.addConstr(X + Y <= R * R, name="containment")

# 7. Solve the model
model.optimize()

# 8. Print results
if model.status == GRB.OPTIMAL:
    L_val = L.X
    W_val = W.X
    A_val = A.X
    print("Optimal solution found:")
    print(f"Length L = {L_val:.4f} meters")
    print(f"Width W = {W_val:.4f} meters")
    print(f"Maximum area A = {A_val:.4f} square meters")
    print(f"FinalAnswer=【{A_val:.4f}】")
else:
    print(f"Optimization failed. Status: {model.status}")
    print(f"FinalAnswer=【0】")