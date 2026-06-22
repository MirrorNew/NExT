import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs
# Price A parameters: [intercept, slope] -> price_A = intercept + slope * x1
price_A = [70.0, -4.0]
# Price B parameters: [intercept, slope] -> price_B = intercept + slope * x2
price_B = [150.0, -15.0]
# Unit production cost in yuan
unit_production_cost_yuan = 150000
# Fixed machine loss fee in yuan
machine_loss_fee_yuan = 1000000

# Convert costs to 10,000 yuan units to match the pricing function units
unit_cost = unit_production_cost_yuan / 10000.0  # 15.0
fixed_cost = machine_loss_fee_yuan / 10000.0   # 100.0

# 2. Create the Gurobi model
model = gp.Model("Profit_Maximization")

# 3. Create decision variables
# Demand quantity sold to Customer A and Customer B
x1 = model.addVar(lb=0, ub=17.5, vtype=GRB.CONTINUOUS, name="x1")
x2 = model.addVar(lb=0, ub=10.0, vtype=GRB.CONTINUOUS, name="x2")
# Binary variable for activating fixed costs
y = model.addVar(vtype=GRB.BINARY, name="y")

# 4. Create auxiliary substitution variables
# For quadratic terms x1^2 and x2^2
x1_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x1_sq")
x2_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x2_sq")

# 5. Set up the objective function
# Profit = x1 * (70 - 4*x1) + x2 * (150 - 15*x2) - 15 * (x1 + x2) - 100 * y
# Simplified: Profit = (70 - 15)*x1 - 4*x1^2 + (150 - 15)*x2 - 15*x2^2 - 100*y
objective = (price_A[0] - unit_cost) * x1 + price_A[1] * x1_sq + \
            (price_B[0] - unit_cost) * x2 + price_B[1] * x2_sq - \
            fixed_cost * y
model.setObjective(objective, GRB.MAXIMIZE)

# 6. Add all constraints
# Link auxiliary variables x1_sq and x2_sq to x1 and x2
model.addGenConstrPow(x1, x1_sq, 2)
model.addGenConstrPow(x2, x2_sq, 2)

# Link indicator variable y to production: y=1 if production occurs, y=0 if not
# Using the instructions to avoid Big-M and use indicator functions
model.addGenConstrIndicator(y, 1, x1 + x2 >= 1e-4)
model.addGenConstrIndicator(y, 0, x1 + x2 <= 0)

# 7. Solve the model and print results
# Set NonConvex parameter to 2 as required for general power constraints/non-convex quadratic problems
model.Params.NonConvex = 2

model.optimize()

# Output the result
if model.status == GRB.OPTIMAL:
    max_profit = model.ObjVal
    print(f"FinalAnswer=【{max_profit}】")
else:
    print("Optimal solution not found.")