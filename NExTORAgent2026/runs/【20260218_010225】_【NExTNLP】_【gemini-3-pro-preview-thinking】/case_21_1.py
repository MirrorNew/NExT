import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and any other necessary packages.
# Defined above.

# 2. Define all parameter matrices and data inputs.
# Using the exact Parameters List provided.
number_of_products = 2
capacity = 300
space_per_unit = [None, 1, 2]  # Index 1 for Prod 1, Index 2 for Prod 2
D = [800, 400]
K = [50, 50]
h = [1, 1]
# Note: D, K, h are 0-indexed lists corresponding to Product 1 and Product 2 respectively.

# 3. Create decision variables.
model = gp.Model("InventoryOptimization")

# Set NonConvex parameter to 2 to handle quadratic equality constraints (Q * InvQ == 1)
model.Params.NonConvex = 2

# Decision Variables Q1, Q2 (Continuous, Q >= 0)
# We use a small lower bound to prevent division by zero in logic, 
# though mathematically handled by the inverse auxiliary variable.
Q = {}
for i in range(number_of_products):
    Q[i] = model.addVar(lb=0.0001, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"Q{i+1}")

# 4. Create any auxiliary substitution variables
# Auxiliary variables InvQ for 1/Q terms. 
# Instruction suggests lb=-GRB.INFINITY for auxiliary vars.
InvQ = {}
for i in range(number_of_products):
    InvQ[i] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"InvQ{i+1}")

# 5. Set up the objective function.
# Minimize Total Annual Cost Z
# Z = Sum( K_i * (D_i / Q_i) + h_i * (Q_i / 2) )
# Substituting 1/Q_i with InvQ_i:
# Z = Sum( K_i * D_i * InvQ_i + 0.5 * h_i * Q_i )

obj_expr = gp.LinExpr()
for i in range(number_of_products):
    # Term 1: Ordering Cost
    obj_expr += K[i] * D[i] * InvQ[i]
    # Term 2: Holding Cost
    obj_expr += 0.5 * h[i] * Q[i]

model.setObjective(obj_expr, GRB.MINIMIZE)

# 6. Add all constraints.

# Constraint A: Auxiliary variable definition (Inverse relationship)
# Q_i * InvQ_i = 1
for i in range(number_of_products):
    model.addConstr(Q[i] * InvQ[i] == 1, name=f"Aux_Inverse_Q{i+1}")

# Constraint B: Storage space capacity
# Sum( space_per_unit_i * (Q_i / 2) ) <= 300
# Product 1 (i=0) uses space_per_unit[1]
# Product 2 (i=1) uses space_per_unit[2]
space_expr = gp.LinExpr()
for i in range(number_of_products):
    s_unit = space_per_unit[i+1]
    average_inventory = Q[i] / 2
    space_expr += s_unit * average_inventory

model.addConstr(space_expr <= capacity, name="Storage_Constraint")

# 7. Solve the model and print results.
model.optimize()

if model.Status == GRB.OPTIMAL:
    print("\nOptimal Solution Found:")
    for i in range(number_of_products):
        print(f"Product {i+1}: Order Quantity Q = {Q[i].X:.4f}")
    print(f"Total Annual Cost: {model.ObjVal:.4f}")
    
    # Output the FinalAnswer as the minimized objective value
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("No optimal solution found.")