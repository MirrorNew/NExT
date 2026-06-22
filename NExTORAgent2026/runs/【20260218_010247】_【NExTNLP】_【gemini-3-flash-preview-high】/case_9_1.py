import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and any other necessary packages.
# Already imported gp and GRB.

# 2. Define all parameter matrices and data inputs.
number_of_plants = 12
x_i = [1.0, 3.0, 0.5, 5.0, 2.0, 4.0, 1.5, 3.5, 0.8, 2.5, 4.5, 1.2]
y_i = [2.0, 1.5, 4.0, 3.0, 2.5, 5.0, 1.0, 4.5, 3.2, 0.5, 2.0, 5.5]
p_i = [50, 80, 30, 100, 60, 70, 40, 90, 55, 75, 85, 65]
safety_distance = 0.8

# Create the model
model = gp.Model("XiostaWarehouseSiteSelection")

# 3. Create decision variables.
# Coordinates of the centralized warehouse
x = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
y = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y")

# 4. Create any auxiliary substitution or indicator variables in coding advice.
# These represent coordinate differences, their squares, and the Euclidean distance.
dx = model.addVars(number_of_plants, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="dx")
dy = model.addVars(number_of_plants, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="dy")
dx2 = model.addVars(number_of_plants, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="dx2")
dy2 = model.addVars(number_of_plants, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="dy2")
dsq = model.addVars(number_of_plants, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="dsq")
d = model.addVars(number_of_plants, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="d")

# 5. Set up the objective function.
# Minimize the total weighted Euclidean distance: ∑ (p_i * d_i)
model.setObjective(gp.quicksum(p_i[i] * d[i] for i in range(number_of_plants)), GRB.MINIMIZE)

# 6. Add all constraints (including gen-constr and indicator constraints).
# Set model to NonConvex to handle quadratic constraints and general constraints defining distance.
model.Params.NonConvex = 2

for i in range(number_of_plants):
    # Coordinate differences: dx[i] = x - x_i, dy[i] = y - y_i
    model.addConstr(dx[i] == x - x_i[i], name=f"dx_constr_{i}")
    model.addConstr(dy[i] == y - y_i[i], name=f"dy_constr_{i}")
    
    # Calculation of squared differences: dx2[i] = dx[i]^2, dy2[i] = dy[i]^2
    model.addGenConstrPow(dx[i], dx2[i], 2, name=f"dx2_pow_{i}")
    model.addGenConstrPow(dy[i], dy2[i], 2, name=f"dy2_pow_{i}")
    
    # Sum of squares for the distance: dsq[i] = dx2[i] + dy2[i]
    model.addConstr(dsq[i] == dx2[i] + dy2[i], name=f"dsq_constr_{i}")
    
    # Calculation of distance via square root: d[i] = dsq[i]^0.5
    model.addGenConstrPow(dsq[i], d[i], 0.5, name=f"d_sqrt_{i}")
    
    # Safety distance constraint: d[i] >= 0.8 km
    model.addConstr(d[i] >= safety_distance, name=f"safety_constr_{i}")

# 7. Solve the model and print results.
model.optimize()

if model.status == GRB.OPTIMAL:
    # Output the optimized objective value (total weighted distance)
    final_answer = model.ObjVal
    print(f"FinalAnswer=【{final_answer}】")
else:
    print("Optimization was not successful.")