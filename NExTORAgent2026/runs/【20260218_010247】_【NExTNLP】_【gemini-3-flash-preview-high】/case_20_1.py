import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and any other necessary packages.

# 2. Define all parameter matrices and data inputs from the provided Parameters List.
m = 2
n = 2
d = [80, 70]
s_max = [100, 100]
cost_quad = [[0.01, 0.01], [0.02, 0.02]]
cost_lin = [[2.0, 3.0], [2.5, 1.5]]

# Create the model
model = gp.Model("RhodeIslandPharmLogistics")

# 3. Create decision variables.
# Range [0, 100] as specified in the context variables.
x = {}
for i in range(m):
    for j in range(n):
        x[i, j] = model.addVar(lb=0.0, ub=100.0, vtype=GRB.CONTINUOUS, name=f"x_{i+1}_{j+1}")

# 4. Create auxiliary substitution variables.
# As per coding advice, these variables range from negative infinity to positive infinity.
y = {}
for i in range(m):
    for j in range(n):
        # Auxiliary variables y_ij = x_ij^2
        y[i, j] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"y_{i+1}_{j+1}")

# Link y[i, j] = x[i, j]^2 using the addGenConstrPow statement as per coding advice.
# Note: model.Params.NonConvex = 2 is required for general constraints like Pow.
model.Params.NonConvex = 2
for i in range(m):
    for j in range(n):
        model.addGenConstrPow(x[i, j], y[i, j], 2)

# 5. Set up the objective function.
# Z = sum(cost_quad[i][j] * x[i,j]^2 + cost_lin[i][j] * x[i,j])
# Substituting x^2 with y
objective = gp.Quicksum(cost_quad[i][j] * y[i, j] + cost_lin[i][j] * x[i, j] for i in range(m) for j in range(n))
model.setObjective(objective, GRB.MINIMIZE)

# 6. Add all constraints.
# Customer demand satisfaction (Total amount received by each customer equals demand)
# Warehouse 1 to Customer 1 + Warehouse 2 to Customer 1 = 80
# Warehouse 1 to Customer 2 + Warehouse 2 to Customer 2 = 70
for j in range(n):
    model.addConstr(gp.Quicksum(x[i, j] for i in range(m)) == d[j], name=f"Demand_Customer_{j+1}")

# Warehouse supply capacity (Total amount sent by each warehouse does not exceed 100)
# x11 + x12 <= 100
# x21 + x22 <= 100
for i in range(m):
    model.addConstr(gp.Quicksum(x[i, j] for j in range(n)) <= s_max[i], name=f"Supply_Warehouse_{i+1}")

# 7. Solve the model and print results.
model.optimize()

if model.Status == GRB.OPTIMAL:
    # Output the minimum transportation cost as requested.
    min_transportation_cost = model.ObjVal
    print(f"FinalAnswer=【{min_transportation_cost}】")