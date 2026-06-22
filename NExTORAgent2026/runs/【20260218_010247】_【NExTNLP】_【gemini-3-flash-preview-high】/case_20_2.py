import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and any other necessary packages.

# 2. Define all parameter matrices and data inputs.
m = 2
n = 2
d = [80, 70]
s_max = [100, 100]
cost_quad = [[0.01, 0.01], [0.02, 0.02]]
cost_lin = [[2.0, 3.0], [2.5, 1.5]]

# Create the model
model = gp.Model("LogisticsCostOptimization")

# 3. Create decision variables.
x = {}
for i in range(m):
    for j in range(n):
        x[i, j] = model.addVar(lb=0.0, ub=100.0, vtype=GRB.CONTINUOUS, name=f"x_{i+1}_{j+1}")

# 4. Create any auxiliary substitution or indicator variables.
# As per instructions: identify expressions like X^2 and use Y = X^2 with addGenConstrPow.
# The values of these auxiliary variables should range from negative infinity to positive infinity.
y = {}
for i in range(m):
    for j in range(n):
        y[i, j] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"y_{i+1}_{j+1}")

# Link y[i, j] = x[i, j]^2 using the addGenConstrPow statement.
# Gurobi requires NonConvex = 2 for general power constraints.
model.Params.NonConvex = 2
for i in range(m):
    for j in range(n):
        # Constraint format: model.addGenConstrPow(xvar, yvar, pow) implies yvar = xvar^pow
        model.addGenConstrPow(x[i, j], y[i, j], 2)

# 5. Set up the objective function.
# Z = sum(cost_quad[i][j] * x[i,j]^2 + cost_lin[i][j] * x[i,j])
# Substituting x^2 with y:
objective = gp.Quicksum(cost_quad[i][j] * y[i, j] + cost_lin[i][j] * x[i, j] for i in range(m) for j in range(n))
model.setObjective(objective, GRB.MINIMIZE)

# 6. Add all constraints (including gen-constr and indicator constraints).

# Customer demand satisfaction (Total amount received by each customer equals demand)
for j in range(n):
    model.addConstr(gp.Quicksum(x[i, j] for i in range(m)) == d[j], name=f"Demand_Customer_{j+1}")

# Warehouse supply capacity (Total amount sent by each warehouse does not exceed max supply)
for i in range(m):
    model.addConstr(gp.Quicksum(x[i, j] for j in range(n)) <= s_max[i], name=f"Supply_Warehouse_{i+1}")

# 7. Solve the model and print results.
model.optimize()

if model.Status == GRB.OPTIMAL:
    # Get the value of the objective function (minimum transportation cost)
    min_cost = model.ObjVal
    print(f"FinalAnswer=【{min_cost}】")