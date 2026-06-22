import gurobipy as gp
from gurobipy import GRB

# 2. Define all parameter matrices and data inputs.
# Parameters List: 
# [{'Name': 'number_of_products', 'Type': 'integer', 'Value': 2}, 
#  {'Name': 'capacity', 'Type': 'integer', 'Value': 300}, 
#  {'Name': 'space_per_unit', 'Type': 'list', 'Value': [None, 1, 2]}, 
#  {'Name': 'D', 'Type': 'list', 'Value': [800, 400]}, 
#  {'Name': 'K', 'Type': 'list', 'Value': [50, 50]}, 
#  {'Name': 'h', 'Type': 'list', 'Value': [1, 1]}, 
#  {'Name': 'beverage_space_ratio', 'Type': 'float', 'Value': 1.5}]

number_of_products = 2
capacity = 300
space_per_unit = [None, 1, 2]
D = [800, 400]
K = [50, 50]
h = [1, 1]
beverage_space_ratio = 1.5

# Create model
model = gp.Model("InventoryOptimization")

# As advised, set NonConvex parameter for solving bilinear constraints (Q * InvQ == 1)
model.Params.NonConvex = 2

# 3. Create decision variables.
# Order quantities for Product 1 and Product 2
Q1 = model.addVar(lb=1e-6, vtype=GRB.CONTINUOUS, name="Q1")
Q2 = model.addVar(lb=1e-6, vtype=GRB.CONTINUOUS, name="Q2")

# 4. Create any auxiliary substitution or indicator variables in coding advice.
# Auxiliary variables for the reciprocals of Q1 and Q2 to handle division.
InvQ1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="InvQ1")
InvQ2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="InvQ2")

# 5. Set up the objective function.
# Z = (K[0] * D[0] / Q1 + h[0] * Q1 / 2) + (K[1] * D[1] / Q2 + h[1] * Q2 / 2)
# Substituting InvQ1 = 1/Q1 and InvQ2 = 1/Q2
obj = (K[0] * D[0] * InvQ1 + 0.5 * h[0] * Q1) + (K[1] * D[1] * InvQ2 + 0.5 * h[1] * Q2)
model.setObjective(obj, GRB.MINIMIZE)

# 6. Add all constraints.
# Relationship between variables and their reciprocals
model.addConstr(Q1 * InvQ1 == 1, name="Reciprocal_Constraint_1")
model.addConstr(Q2 * InvQ2 == 1, name="Reciprocal_Constraint_2")

# Storage space capacity: sum of average inventory space <= capacity
# average inventory space for product i is space_per_unit[i] * (Qi / 2)
model.addConstr(0.5 * space_per_unit[1] * Q1 + 0.5 * space_per_unit[2] * Q2 <= capacity, name="StorageCapacity")

# 7. Solve the model and print results.
model.optimize()

if model.status == GRB.OPTIMAL:
    final_cost = model.ObjVal
    print(f"Optimal Order Quantity Q1: {Q1.X}")
    print(f"Optimal Order Quantity Q2: {Q2.X}")
    print(f"Minimum Total Annual Cost: {final_cost}")
    print(f"FinalAnswer=【{final_cost}】")