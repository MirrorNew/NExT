import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs
# Given parameters
alpha = 0.3
beta = 0.8
p_x = 4
p_y = 2
M = 100
N_threshold = 30
tip_rate = 0.1

# 2. Create the model
model = gp.Model("Utility_Optimization")
# Identify any function expressions that require auxiliary substitution variables, and use "model.Params.NonConvex = 2"
model.Params.NonConvex = 2

# 3. Create decision variables
# Commodity quantities must be integers based on the problem description
x = model.addVar(vtype=GRB.INTEGER, lb=0, ub=25, name="x")
y = model.addVar(vtype=GRB.INTEGER, lb=0, ub=50, name="y")
# Indicator variable for service fee application
z = model.addVar(vtype=GRB.BINARY, name="z")

# 4. Create auxiliary substitution variables
# These variables should range from negative infinity to positive infinity
v1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="v1")
v2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="v2")
U = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="U")

# 5. Set up the objective function
# Goal is to maximize utility U(x, y)
model.setObjective(U, GRB.MAXIMIZE)

# 6. Add all constraints
# Utility function components: v1 = x^0.3, v2 = y^0.8
# Power constraint Y = X^a: model.addGenConstrPow(X, Y, a)
model.addGenConstrPow(x, v1, alpha)
model.addGenConstrPow(y, v2, beta)

# Utility substitution: U = v1 * v2
# Using standard bilinear multiplication constraint
model.addConstr(v1 * v2 == U)

# Service-fee logic (Indicator-variable scenarios):
# x + y > 30 implies z = 1 (Since x, y are integers, > 30 is >= 31)
model.addGenConstrIndicator(z, 1, x + y >= N_threshold + 1)
model.addGenConstrIndicator(z, 0, x + y <= N_threshold)

# Budget constraints based on indicator z
# If z=1: total price + 0.1 * total price <= M  => 1.1 * (4x + 2y) <= 100
model.addGenConstrIndicator(z, 1, (1 + tip_rate) * p_x * x + (1 + tip_rate) * p_y * y <= M)
# If z=0: total price <= M => 4x + 2y <= 100
model.addGenConstrIndicator(z, 0, p_x * x + p_y * y <= M)

# 7. Solve the model and print results
model.optimize()

if model.Status == GRB.OPTIMAL:
    max_utility = U.X
    print(f"Optimal quantity x: {x.X}")
    print(f"Optimal quantity y: {y.X}")
    print(f"Maximum Utility: {max_utility}")
    # ATTENTION 1: Output the final answer in the required format
    print(f"FinalAnswer=【{max_utility}】")
else:
    print("Optimal solution not found.")