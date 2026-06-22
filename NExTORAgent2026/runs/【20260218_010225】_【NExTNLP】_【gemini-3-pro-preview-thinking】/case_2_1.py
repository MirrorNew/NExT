import gurobipy as gp
from gurobipy import GRB

# 1. Create model
model = gp.Model("Bridge_Beam_Design")

# Set NonConvex parameter to 2 to handle non-convex quadratic constraints (e.g., Z = x1 * Y)
model.Params.NonConvex = 2

# 2. Define parameter matrices and data inputs
r = 300  # Radius of the circular steel billet

# 3. Create decision variables
# x1: half-width of the rectangular cross-section
# x2: half-height of the rectangular cross-section
x1 = model.addVar(lb=0.0, ub=r, vtype=GRB.CONTINUOUS, name="x1")
x2 = model.addVar(lb=0.0, ub=r, vtype=GRB.CONTINUOUS, name="x2")

# 4. Create auxiliary substitution variables
# Instructions specify these should have infinite bounds
# Y represents x2^2
Y = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Y")
# Z represents x1 * x2^2 (the objective kernel)
Z = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Z")

# 5. Set up the objective function
# Maximize the bending moment, which is proportional to x1 * x2^2
model.setObjective(Z, GRB.MAXIMIZE)

# 6. Add all constraints
# Geometric Constraint: The rectangle must fit within the circular billet (x1^2 + x2^2 <= r^2)
model.addConstr(x1**2 + x2**2 <= r**2, name="Circular_Boundary")

# Auxiliary Constraint for Y = x2^2
# Using addGenConstrPow as requested for power functions
model.addGenConstrPow(x2, Y, 2, name="Power_Constraint_Y")

# Auxiliary Constraint for Z = x1 * Y (Substitution for x1 * x2^2)
# Using direct multiplication constraint as requested
model.addConstr(x1 * Y == Z, name="Mult_Constraint_Z")

# 7. Solve the model and print results
model.optimize()

if model.status == GRB.OPTIMAL:
    print("Optimization Successful")
    print(f"Optimal Half-Width (x1): {x1.X}")
    print(f"Optimal Half-Height (x2): {x2.X}")
    print(f"Objective Value (Z): {Z.X}")
    
    # Output the answer to the question: give the value of x1
    print(f"FinalAnswer=【{x1.X}】")
else:
    print("Optimization failed or no solution found.")