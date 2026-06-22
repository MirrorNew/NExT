import gurobipy as gp
from gurobipy import GRB

# Define parameters from the provided list
num_products = 2
capacity = 300
space_per_unit = [None, 1, 2]  # Indexed by product number (1-based)
D = [800, 400]  # Annual demand
K = [50, 50]    # Fixed order cost
h = [1, 1]      # Unit annual holding cost

# Create model
model = gp.Model("EOQ_Inventory_Optimization")

# Create decision variables
Q1 = model.addVar(lb=0, name="Q1")
Q2 = model.addVar(lb=0, name="Q2")

# Create auxiliary substitution variables for 1/Q1 and 1/Q2
R1 = model.addVar(lb=0, ub=GRB.INFINITY, name="R1")
R2 = model.addVar(lb=0, ub=GRB.INFINITY, name="R2")

# Set objective function
# Z = K1 * D1 * (1/Q1) + h1 * (Q1/2) + K2 * D2 * (1/Q2) + h2 * (Q2/2)
# Using auxiliary variables: Z = K1 * D1 * R1 + h1 * (Q1/2) + K2 * D2 * R2 + h2 * (Q2/2)
obj_expr = K[0] * D[0] * R1 + h[0] * Q1 / 2 + K[1] * D[1] * R2 + h[1] * Q2 / 2
model.setObjective(obj_expr, GRB.MINIMIZE)

# Add constraints for auxiliary variables: Q1 * R1 = 1 and Q2 * R2 = 1
model.addConstr(Q1 * R1 == 1, name="aux_constraint1")
model.addConstr(Q2 * R2 == 1, name="aux_constraint2")

# Add storage space constraint: (space1 * Q1/2) + (space2 * Q2/2) ≤ capacity
# space1 = 1, space2 = 2
space_constraint = (space_per_unit[1] * Q1 / 2) + (space_per_unit[2] * Q2 / 2) <= capacity
model.addConstr(space_constraint, name="storage_space")

# Set non-convex parameter since we have bilinear constraints Q1*R1=1 and Q2*R2=1
model.Params.NonConvex = 2

# Optimize the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    print(f"Q1 = {Q1.X:.2f}")
    print(f"Q2 = {Q2.X:.2f}")
    print(f"Total annual cost = {model.ObjVal:.2f} yuan")
    
    # Output the answer to the question (order quantities)
    print(f"FinalAnswer=【Q1={Q1.X:.2f}, Q2={Q2.X:.2f}】")
else:
    print(f"Optimization failed. Status: {model.status}")
    print(f"FinalAnswer=【No optimal solution found】")