import gurobipy as gp
from gurobipy import GRB

# Create the Gurobi model
model = gp.Model("Fountain_Pool_Optimization")

# Set the NonConvex parameter to 2 to allow non-convex quadratic objective (Maximizing L*W)
model.Params.NonConvex = 2

# 2. Define all parameter matrices and data inputs.
R = 10  # Radius of the circular square in meters

# 3. Create decision variables.
L = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="L")
W = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="W")

# 4. Create any auxiliary substitution or indicator variables in coding advice.
# In this specific problem, Gurobi handles the quadratic terms L*W and L*L natively 
# with the NonConvex=2 parameter, so explicit auxiliary variables for powers are not strictly necessary 
# if using standard quadratic constraints. The logic follows the provided math advice.

# 5. Set up the objective function.
# Objective: Maximize the area of the rectangular pool A = L * W
model.setObjective(L * W, GRB.MAXIMIZE)

# 6. Add all constraints.
# Containment Constraint: (L/2)^2 + (W/2)^2 <= R^2
# This ensures the rectangle is inscribed within the circle.
# Rewritten as 0.25 * L * L + 0.25 * W * W <= R^2 to strictly follow quadratic form syntax
model.addConstr(0.25 * L * L + 0.25 * W * W <= R**2, name="Containment")

# 7. Solve the model and print results.
model.optimize()

if model.Status == GRB.OPTIMAL:
    print(f"Optimal Length L: {L.X}")
    print(f"Optimal Width W: {W.X}")
    print(f"Maximum Area: {model.ObjVal}")
    # ATTENTION 1: FinalAnswer output
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimization was not successful.")