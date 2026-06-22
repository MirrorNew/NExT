import gurobipy as gp
from gurobipy import GRB

# 2. Define all parameter matrices and data inputs
m = 2  # number of warehouses
n = 2  # number of customers
d = [80, 70]  # demand for customers
s_max = [100, 100]  # maximum supply for warehouses
cost_quad = [[0.01, 0.01], [0.02, 0.02]]  # quadratic cost coefficients
cost_lin = [[2.0, 3.0], [2.5, 1.5]]  # linear cost coefficients

# 3. Create decision variables
model = gp.Model("TransportationCostOptimization")

# Create shipment variables x[i][j]
x = {}
for i in range(m):
    for j in range(n):
        x[i, j] = model.addVar(lb=0, ub=100, name=f"x_{i+1}{j+1}")

# 4. Create auxiliary substitution variables for squared terms
y = {}
for i in range(m):
    for j in range(n):
        y[i, j] = model.addVar(lb=0, ub=GRB.INFINITY, name=f"y_{i+1}{j+1}")

# Set non-convex parameter
model.Params.NonConvex = 2

# 5. Set up the objective function
obj_expr = gp.QuadExpr()
for i in range(m):
    for j in range(n):
        obj_expr += cost_quad[i][j] * y[i, j] + cost_lin[i][j] * x[i, j]
model.setObjective(obj_expr, GRB.MINIMIZE)

# 6. Add all constraints
# Power constraints for squared terms
for i in range(m):
    for j in range(n):
        model.addGenConstrPow(x[i, j], y[i, j], 2, name=f"pow_constr_{i+1}{j+1}")

# Demand constraints
for j in range(n):
    demand_expr = gp.LinExpr()
    for i in range(m):
        demand_expr += x[i, j]
    model.addConstr(demand_expr == d[j], name=f"demand_customer_{j+1}")

# Supply constraints
for i in range(m):
    supply_expr = gp.LinExpr()
    for j in range(n):
        supply_expr += x[i, j]
    model.addConstr(supply_expr <= s_max[i], name=f"supply_warehouse_{i+1}")

# 7. Solve the model and print results
model.optimize()

if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    print("\nShipment volumes:")
    for i in range(m):
        for j in range(n):
            print(f"x_{i+1}{j+1} = {x[i, j].X:.2f}")
    
    total_cost = model.ObjVal
    print(f"\nMinimum transportation cost: {total_cost:.2f}")
    print(f"FinalAnswer=【{total_cost:.2f}】")
else:
    print(f"Optimization failed. Status: {model.status}")
    print(f"FinalAnswer=【Optimization failed】")