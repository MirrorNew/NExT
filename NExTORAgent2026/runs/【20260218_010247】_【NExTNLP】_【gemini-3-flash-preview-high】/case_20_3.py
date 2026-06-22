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
model = gp.Model("KazdaleLogisticsOptimization")

# 3. Create decision variables.
x = {}
for i in range(m):
    for j in range(n):
        x[i, j] = model.addVar(lb=0.0, ub=100.0, vtype=GRB.CONTINUOUS, name=f"x_{i+1}_{j+1}")

# 4. Create auxiliary substitution variables.
# For X^2, we need Y = X^2.
y = {}
for i in range(m):
    for j in range(n):
        # Per instruction, range for auxiliary variables should be negative infinity to positive infinity.
        y[i, j] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"y_{i+1}_{j+1}")

# Set up the objective function.
# Z = sum(cost_quad[i][j] * y[i,j] + cost_lin[i][j] * x[i,j])
objective = gp.quicksum(cost_quad[i][j] * y[i, j] + cost_lin[i][j] * x[i, j] for i in range(m) for j in range(n))
model.setObjective(objective, GRB.MINIMIZE)

# 6. Add all constraints (including gen-constr and indicator constraints).

# Set NonConvex parameter to 2 as required for general constraints like Pow.
model.Params.NonConvex = 2

# Add General Power constraints for the squared terms.
for i in range(m):
    for j in range(n):
        # Format: model.addGenConstrPow(x_var, y_var, exponent) => y_var = x_var^exponent
        model.addGenConstrPow(x[i, j], y[i, j], 2)

# Customer demand constraints: sum of volumes to customer j equals demand d[j].
for j in range(n):
    model.addConstr(gp.quicksum(x[i, j] for i in range(m)) == d[j], name=f"Demand_Customer_{j+1}")

# Warehouse supply constraints: sum of volumes from warehouse i does not exceed s_max[i].
for i in range(m):
    model.addConstr(gp.quicksum(x[i, j] for j in range(n)) <= s_max[i], name=f"Supply_Warehouse_{i+1}")

# 7. Solve the model and print results.
model.optimize()

if model.Status == GRB.OPTIMAL:
    result_val = model.ObjVal
    print(f"FinalAnswer=【{result_val}】")