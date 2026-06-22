import gurobipy as gp
from gurobipy import GRB

# Define parameter matrices and data inputs
demand_intercept = 100
demand_slope = -5
unit_production_cost = 10
price_exponent = 0.9
price_lower_bound = 0
price_upper_bound = 20

# Create model
model = gp.Model("DrugPricingOptimization")

# Set global parameter to handle non-convexity (bilinear and power functions)
model.Params.NonConvex = 2

# Create decision variables
p = model.addVar(lb=price_lower_bound, ub=price_upper_bound, vtype=GRB.CONTINUOUS, name="p")
D = model.addVar(lb=0, ub=demand_intercept, vtype=GRB.CONTINUOUS, name="D")

# Create auxiliary substitution variables
# The instruction says values range from negative infinity to positive infinity
p_pow_09 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="p_pow_09")
p09_D = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="p09_D")

# Set up the objective function: Maximize Π = (p^0.9 - 10) * D = p^0.9 * D - 10 * D
model.setObjective(p09_D - unit_production_cost * D, GRB.MAXIMIZE)

# Add constraints
# Demand Function: D = 100 - 5p
model.addConstr(D == demand_intercept + demand_slope * p, name="Demand_Function")

# Non-linear relationship: p_pow_09 = p^0.9
# addGenConstrPow(x, y, a) => y = x^a
model.addGenConstrPow(p, p_pow_09, price_exponent, name="Power_Constraint")

# Bilinear relationship: p09_D = p_pow_09 * D
model.addConstr(p09_D == p_pow_09 * D, name="Bilinear_Constraint")

# Solve the model
model.optimize()

# Print results
if model.Status == GRB.OPTIMAL:
    max_profit = model.ObjVal
    optimal_price = p.X
    print(f"Optimal Price: {optimal_price:.4f} yuan")
    print(f"Maximum Profit: {max_profit:.4f}")
    print(f"FinalAnswer=【{max_profit}】")
else:
    print("Optimal solution not found.")