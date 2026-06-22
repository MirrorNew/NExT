import gurobipy as gp
from gurobipy import GRB

# Define parameters from the provided list
demand_intercept = 100
demand_slope = -5
unit_production_cost = 10
price_exponent = 0.9
price_lower_bound = 0
price_upper_bound = 20

# Create model
model = gp.Model("PharmaceuticalPricing")

# Create decision variables
p = model.addVar(lb=price_lower_bound, ub=price_upper_bound, vtype=GRB.CONTINUOUS, name="p")
D = model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS, name="D")

# Create auxiliary variables (unbounded as advised)
Y_power = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Y_power")
profit_per_unit = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="profit_per_unit")
total_profit = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="total_profit")

# Set non-convex parameter for power constraint
model.Params.NonConvex = 2

# Add constraints
# 1. Y_power = p^0.9
model.addGenConstrPow(p, Y_power, price_exponent, name="power_constraint")

# 2. profit_per_unit = p^0.9 - 10 = Y_power - 10
model.addConstr(profit_per_unit == Y_power - unit_production_cost, name="profit_per_unit_def")

# 3. Demand function: D = 100 - 5p
model.addConstr(D == demand_intercept + demand_slope * p, name="demand_function")

# 4. total_profit = profit_per_unit * D
model.addConstr(total_profit == profit_per_unit * D, name="total_profit_def")

# Set objective: maximize total_profit
model.setObjective(total_profit, GRB.MAXIMIZE)

# Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print(f"Optimal price p = {p.X:.4f}")
    print(f"Demand D = {D.X:.4f}")
    print(f"Maximum profit Π = {total_profit.X:.4f}")
    print(f"FinalAnswer=【{total_profit.X:.4f}】")
else:
    print(f"Optimization failed. Status: {model.status}")
    print(f"FinalAnswer=【Optimization failed】")