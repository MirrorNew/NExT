import gurobipy as gp
from gurobipy import GRB
import math

# 1. Initialize Model
model = gp.Model("HazardousWarehouseLocation")
model.Params.NonConvex = 2  # Allow non-convex quadratic constraints

# 2. Parameters and Data
# Using the values from the Parameters List
number_of_companies = 12
number_of_plants = 12
I_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
x_coords = [1.0, 3.0, 0.5, 5.0, 2.0, 4.0, 1.5, 3.5, 0.8, 2.5, 4.5, 1.2]
y_coords = [2.0, 1.5, 4.0, 3.0, 2.5, 5.0, 1.0, 4.5, 3.2, 0.5, 2.0, 5.5]
p_counts = [50, 80, 30, 100, 60, 70, 40, 90, 55, 75, 85, 65]
safety_distance = 0.8

# Ensure indices match (Python is 0-indexed, data I is 1-based, but lists are ordered)
# We will iterate 0 to 11

# 3. Decision Variables
# Coordinates of the centralized warehouse (x, y). They can be anywhere, so bounds are -inf to +inf.
x = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
y = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y")

# 4. Auxiliary Variables
# dx[i], dy[i] representing the difference in coordinates for each factory i
dx = {}
dy = {}
# d[i] representing the Euclidean distance for each factory i
d = {}

for i in range(number_of_plants):
    # Differences can be negative
    dx[i] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"dx_{i}")
    dy[i] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"dy_{i}")
    # Distance is non-negative
    d[i] = model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"d_{i}")

# 5. Constraints
for i in range(number_of_plants):
    # Definition of coordinate differences
    model.addConstr(dx[i] == x - x_coords[i], name=f"def_dx_{i}")
    model.addConstr(dy[i] == y - y_coords[i], name=f"def_dy_{i}")
    
    # Definition of Euclidean distance using General Constraint Norm
    # d[i] = sqrt(dx[i]^2 + dy[i]^2)
    model.addGenConstrNorm(d[i], [dx[i], dy[i]], 2.0, name=f"norm_dist_{i}")
    
    # Safety distance constraint: distance >= 0.8 km
    # This creates a non-convex feasible region (holes around factories)
    model.addConstr(d[i] >= safety_distance, name=f"safety_dist_{i}")

# 6. Objective Function
# Minimize sum(p_i * d_i)
obj_expr = gp.quicksum(p_counts[i] * d[i] for i in range(number_of_plants))
model.setObjective(obj_expr, GRB.MINIMIZE)

# 7. Solve and Print
model.optimize()

if model.Status == GRB.OPTIMAL:
    print("Optimization Successful")
    print(f"Optimal Warehouse Location: x = {x.X:.4f}, y = {y.X:.4f}")
    print(f"Minimum Weighted Distance: {model.ObjVal:.4f}")
    
    # Output the required FinalAnswer
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimization failed or not optimal")
    # In case of failure, usually means infeasible or unbounded, but problem structure implies feasible solution exists.
    # We provide output if feasible solution found.
    if model.SolCount > 0:
         print(f"FinalAnswer=【{model.ObjVal}】")
    else:
         print("FinalAnswer=【No Feasible Solution Found】")