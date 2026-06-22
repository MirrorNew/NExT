import gurobipy as gp
from gurobipy import GRB

# 1. Create the model
model = gp.Model("Transportation_Optimization")

# 2. Define Parameters
m = 2  # Number of warehouses
n = 2  # Number of customers
d = [80, 70]  # Demand for Customer 1 and Customer 2
s_max = [100, 100]  # Supply capacity for Warehouse 1 and Warehouse 2
cost_quad = [[0.01, 0.01], [0.02, 0.02]]  # Quadratic cost coefficients
cost_lin = [[2.0, 3.0], [2.5, 1.5]]  # Linear cost coefficients

# Set NonConvex parameter as advised for general constraints involving powers
model.Params.NonConvex = 2

# 3. Create Decision Variables
x = {}
for i in range(m):
    for j in range(n):
        # x_ij: shipment volume from warehouse i to customer j
        x[i, j] = model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS, name=f"x_{i+1}_{j+1}")

# 4. Create Auxiliary Substitution Variables for Quadratic Terms
# We need sq_x_ij = x_ij^2
sq_x = {}
for i in range(m):
    for j in range(n):
        sq_x[i, j] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"sq_x_{i+1}_{j+1}")

# Add General Constraints for powers: sq_x[i,j] = x[i,j]^2
for i in range(m):
    for j in range(n):
        # Note: Order is (InputVar, OutputVar, Exponent) -> OutputVar = InputVar ^ Exponent
        model.addGenConstrPow(x[i, j], sq_x[i, j], 2, name=f"gc_pow_{i+1}_{j+1}")

# 5. Set up the Objective Function
# Minimize Z = sum( cost_quad * x^2 + cost_lin * x )
objective_expr = gp.LinExpr()
for i in range(m):
    for j in range(n):
        objective_expr += cost_quad[i][j] * sq_x[i, j] + cost_lin[i][j] * x[i, j]

model.setObjective(objective_expr, GRB.MINIMIZE)

# 6. Add Constraints

# Constraint 1 & 2: Customer Demand
# x11 + x21 = 80
# x12 + x22 = 70
for j in range(n):
    model.addConstr(gp.quicksum(x[i, j] for i in range(m)) == d[j], name=f"Demand_Customer{j+1}")

# Constraint 3 & 4: Warehouse Supply
# x11 + x12 <= 100
# x21 + x22 <= 100
for i in range(m):
    model.addConstr(gp.quicksum(x[i, j] for j in range(n)) <= s_max[i], name=f"Supply_Warehouse{i+1}")

# 7. Solve the model
model.optimize()

# Print Results
if model.status == GRB.OPTIMAL:
    print("Optimal Solution Found:")
    for i in range(m):
        for j in range(n):
            print(f"x_{i+1}_{j+1} = {x[i, j].X}")
    print(f"Total Cost: {model.objVal}")
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found.")