import gurobipy as gp
from gurobipy import GRB

# ============================
# 1. Define parameters
# ============================
R = 10  # given: radius of circular square (integer, value 10)

# ============================
# 2. Create model
# ============================
model = gp.Model("MaxInscribedRectangleInCircle")

# Allow nonconvex (bilinear/quadratic) expressions
model.Params.NonConvex = 2

# ============================
# 3. Decision variables
# ============================
# Length and width of the rectangular pool
L = model.addVar(name="L", lb=0.0)
W = model.addVar(name="W", lb=0.0)

# ============================
# 4. Auxiliary variables
# ============================
# Area
A = model.addVar(name="A", lb=0.0)

# Squares of L and W
L2 = model.addVar(name="L2", lb=0.0)
W2 = model.addVar(name="W2", lb=0.0)

# Define squares via general constraints: L2 = L^2, W2 = W^2
model.addGenConstrPow(L, L2, 2.0, name="L_square")
model.addGenConstrPow(W, W2, 2.0, name="W_square")

# Link area: A = L * W (bilinear)
model.addConstr(A == L * W, name="Area_def")

# ============================
# 5. Objective function
# ============================
# Maximize area A
model.setObjective(A, GRB.MAXIMIZE)

# ============================
# 6. Constraints
# ============================
# Containment within the circle: (L/2)^2 + (W/2)^2 <= R^2
# Using auxiliary variables: 0.25 * L2 + 0.25 * W2 <= R^2
model.addConstr(0.25 * L2 + 0.25 * W2 <= R * R, name="Containment")

# Non-negativity (already via lb=0, but we add explicit constraints as instructed)
model.addConstr(L >= 0.0, name="L_nonneg")
model.addConstr(W >= 0.0, name="W_nonneg")
model.addConstr(A >= 0.0, name="A_nonneg")
model.addConstr(L2 >= 0.0, name="L2_nonneg")
model.addConstr(W2 >= 0.0, name="W2_nonneg")

# ============================
# 7. Solve the model
# ============================
model.optimize()

# ============================
# 8. Print results
# ============================
if model.Status == GRB.OPTIMAL or model.Status == GRB.INTERRUPTED:
    L_opt = L.X
    W_opt = W.X
    A_opt = A.X

    print(f"Optimal length L: {L_opt:.6f} m")
    print(f"Optimal width  W: {W_opt:.6f} m")
    print(f"Maximum area A: {A_opt:.6f} m^2")

    # Final answer required by instructions:
    # The question asks: "determine the optimal size (length and width) of the rectangular pool and give the maximum area."
    # Here we output all three values as a tuple in the required format.
    print(f"FinalAnswer=【L={L_opt:.6f}, W={W_opt:.6f}, A={A_opt:.6f}】")
else:
    print("No optimal solution found.")
    print("FinalAnswer=【No feasible solution】")