import gurobipy as gp
from gurobipy import GRB

# 1. Initialize Model
model = gp.Model("ConcreteMixingStation")

# 2. Parameters
# Defined strictly from the provided Parameters List
number_of_sections = 4
number_of_work_sites = 4
Q = [10, 15, 20, 25]
coordinates = [[1, 1], [1, 2], [2, 1], [2, 2]]

# 3. Decision Variables
# x and y coordinates of the central mixing station (Unrestricted, continuous)
x = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
y = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y")

# Auxiliary variables
# d[i] represents the Euclidean distance from the station to work site i
d = model.addVars(number_of_work_sites, lb=0.0, vtype=GRB.CONTINUOUS, name="d")

# dx[i] and dy[i] represent the coordinate differences (x - x_i) and (y - y_i)
dx = model.addVars(number_of_work_sites, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="dx")
dy = model.addVars(number_of_work_sites, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="dy")

# 4. Constraints
for i in range(number_of_work_sites):
    x_i = coordinates[i][0]
    y_i = coordinates[i][1]
    
    # Linear constraints to define the differences
    model.addConstr(dx[i] == x - x_i, name=f"link_dx_{i}")
    model.addConstr(dy[i] == y - y_i, name=f"link_dy_{i}")
    
    # Second-Order Cone constraints for Euclidean distance: d[i] = norm(dx[i], dy[i])
    # This corresponds to d[i] >= sqrt(dx[i]^2 + dy[i]^2)
    model.addGenConstrNorm(d[i], [dx[i], dy[i]], 2.0, name=f"norm_constraint_{i}")

# 5. Objective Function
# Minimize the total transportation volume (weighted sum of distances)
obj_expr = gp.quicksum(Q[i] * d[i] for i in range(number_of_work_sites))
model.setObjective(obj_expr, GRB.MINIMIZE)

# 6. Solve the model
model.optimize()

# 7. Output results
if model.Status == GRB.OPTIMAL:
    print(f"Optimal x coordinate: {x.X}")
    print(f"Optimal y coordinate: {y.X}")
    print(f"Minimum Total Transportation Volume: {model.ObjVal}")
    
    # Final Answer format
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimization was not successful.")