import gurobipy as gp

# Data from Parameters List
number_of_plants = 12
I = list(range(number_of_plants))  # 0-based indexing for Python
x_i = [1.0, 3.0, 0.5, 5.0, 2.0, 4.0, 1.5, 3.5, 0.8, 2.5, 4.5, 1.2]
y_i = [2.0, 1.5, 4.0, 3.0, 2.5, 5.0, 1.0, 4.5, 3.2, 0.5, 2.0, 5.5]
p_i = [50, 80, 30, 100, 60, 70, 40, 90, 55, 75, 85, 65]
safety_distance = 0.8

# Create model
model = gp.Model("XiostaWarehouseLocation")

# Decision variables
x = model.addVar(lb=-gp.GRB.INFINITY, name="x")
y = model.addVar(lb=-gp.GRB.INFINITY, name="y")

# Auxiliary variables
# sq_i = (x - x_i)^2 + (y - y_i)^2
sq = model.addVars(I, lb=0, name="sq")
# d_i = sqrt(sq_i)
d = model.addVars(I, lb=0, name="d")

# Set non-convex parameter
model.Params.NonConvex = 2

# Constraints
# Define squared distances
for i in I:
    model.addConstr(sq[i] == (x - x_i[i]) * (x - x_i[i]) + (y - y_i[i]) * (y - y_i[i]), 
                    name=f"sq_dist_{i+1}")

# Define distances via power constraint (square root)
for i in I:
    model.addGenConstrPow(sq[i], d[i], 0.5, name=f"sqrt_{i+1}")

# Safety distance constraints
for i in I:
    model.addConstr(d[i] >= safety_distance, name=f"safety_{i+1}")

# Objective: minimize weighted sum of distances
obj_expr = gp.quicksum(p_i[i] * d[i] for i in I)
model.setObjective(obj_expr, gp.GRB.MINIMIZE)

# Solve
model.optimize()

# Output results
if model.status == gp.GRB.OPTIMAL:
    print(f"Optimal warehouse location:")
    print(f"  x = {x.X:.4f} km")
    print(f"  y = {y.X:.4f} km")
    print(f"Minimum weighted distance sum = {model.ObjVal:.4f}")
    
    # Compute the actual objective value from optimal distances
    weighted_sum = 0
    for i in I:
        dist = ((x.X - x_i[i])**2 + (y.X - y_i[i])**2)**0.5
        weighted_sum += p_i[i] * dist
    print(f"Verified weighted sum = {weighted_sum:.4f}")
    
    # Output the answer as required (the objective value)
    print(f"FinalAnswer=【{model.ObjVal:.4f}】")
else:
    print("No optimal solution found.")
    print(f"FinalAnswer=【None】")