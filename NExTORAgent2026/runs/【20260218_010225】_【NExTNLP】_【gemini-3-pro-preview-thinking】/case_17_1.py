import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
load_demand = 800
a = [5000, 3000, 1000]
b = [5, 3, 8]
c = [0.01, 0.02, 0.005]
P_min = [100, 50, 50]
P_max = [500, 300, 300]

# 2. Create Model
model = gp.Model("PowerDispatch")

# 3. Create decision variables
# P[0] corresponds to P1, P[1] to P2, P[2] to P3
P = []
for i in range(3):
    P.append(model.addVar(lb=P_min[i], ub=P_max[i], vtype=GRB.CONTINUOUS, name=f"P{i+1}"))

# 4. Set up the objective function
# Minimize F = sum(a_i + b_i * P_i + c_i * P_i^2)
# Since the objective is a convex quadratic function (c_i > 0), we can define it directly.
obj_expr = 0
for i in range(3):
    obj_expr += a[i] + b[i] * P[i] + c[i] * P[i] * P[i]

model.setObjective(obj_expr, GRB.MINIMIZE)

# 5. Add all constraints
# Load balance constraint: P1 + P2 + P3 = 800
model.addConstr(gp.quicksum(P) == load_demand, name="LoadBalance")

# 6. Solve the model and print results
model.optimize()

if model.Status == GRB.OPTIMAL:
    print("Optimal Solution Found:")
    for v in model.getVars():
        print(f"{v.VarName} = {v.X}")
    print(f"Total Fuel Cost = {model.ObjVal}")
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("No optimal solution found.")