import gurobipy as gp
from gurobipy import GRB

# 1. Initialize Model
model = gp.Model("Consumer_Utility_Optimization")

# 2. Parameters
alpha = 0.3
beta = 0.8
p_x = 4
p_y = 2
M_budget = 100
N_threshold = 30
tip_rate = 0.1

# 3. Decision Variables
# x: units of commodity x, integer, 0 <= x <= 25
x = model.addVar(vtype=GRB.INTEGER, lb=0, ub=25, name="x")
# y: units of commodity y, integer, 0 <= y <= 50
y = model.addVar(vtype=GRB.INTEGER, lb=0, ub=50, name="y")
# z: indicator variable, 1 if service fee applies (x + y >= 31), 0 otherwise
z = model.addVar(vtype=GRB.BINARY, name="z")
# u: auxiliary fee-base variable
u = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=100, name="u")

# Auxiliary variables for objective function calculation
# aux_x = x^0.3
aux_x = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="aux_x")
# aux_y = y^0.8
aux_y = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="aux_y")
# utility = aux_x * aux_y
utility = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="utility")

# 4. Set Gurobi parameter to handle non-convex quadratic/power constraints
model.Params.NonConvex = 2

# 5. Objective Function
# Maximize Utility = x^0.3 * y^0.8
model.setObjective(utility, GRB.MAXIMIZE)

# 6. Constraints

# Power constraints
# aux_x = x ^ 0.3
model.addGenConstrPow(x, aux_x, alpha, "pow_x")
# aux_y = y ^ 0.8
model.addGenConstrPow(y, aux_y, beta, "pow_y")

# Utility definition constraint (Product)
model.addConstr(utility == aux_x * aux_y, "utility_calc")

# Service Fee Indicator Logic
# If z=1, then x + y >= 31
model.addGenConstrIndicator(z, 1, x + y >= N_threshold + 1, name="ind_z_1")
# If z=0, then x + y <= 30
model.addGenConstrIndicator(z, 0, x + y <= N_threshold, name="ind_z_0")

# Auxiliary Fee Base Logic
# If z=1, u = 4x + 2y (the base cost)
model.addGenConstrIndicator(z, 1, u == p_x * x + p_y * y, name="ind_u_1")
# If z=0, u = 0
model.addGenConstrIndicator(z, 0, u == 0, name="ind_u_0")

# Budget Constraint
# Base Cost + Tip <= Budget
# 4x + 2y + 0.1 * u <= 100
model.addConstr(p_x * x + p_y * y + tip_rate * u <= M_budget, "Budget_Constraint")

# 7. Solve
model.optimize()

# 8. Print Results
if model.status == GRB.OPTIMAL:
    print(f"Optimal Solution Found:")
    print(f"x = {x.X}")
    print(f"y = {y.X}")
    print(f"z = {z.X}")
    print(f"Total Quantity = {x.X + y.X}")
    print(f"Total Cost = {p_x * x.X + p_y * y.X + tip_rate * u.X}")
    print(f"Objective Utility = {model.ObjVal}")
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("No optimal solution found.")