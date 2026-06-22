import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and any other necessary packages.
# (Standard imports included)

# 2. Define all parameter matrices and data inputs.
demand_intercept = 100
demand_slope = -5
unit_production_cost = 10
price_exponent = 0.9
price_lower_bound = 0
price_upper_bound = 20

# Create the model
model = gp.Model("Rhode_Island_Pharmaceuticals_Pricing")

# Set NonConvex parameter to 2 to handle the nonlinear objective (product of variables) and power constraint
model.Params.NonConvex = 2

# 3. Create decision variables.
# p: unit price of the drug, constrained by lower and upper bounds
p = model.addVar(lb=price_lower_bound, ub=price_upper_bound, vtype=GRB.CONTINUOUS, name="p")

# 4. Create any auxiliary substitution or indicator variables.
# D: Market demand. Range set to infinite as per coding advice, though physically limited by constraints.
D = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="D")

# p_pow: Auxiliary variable for the non-linear term p^0.9
p_pow = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="p_pow")

# 5. Set up the objective function.
# Profit = (p^0.9 - 10) * D
# Substituting p^0.9 with p_pow
# Objective: Maximize (p_pow - unit_production_cost) * D
model.setObjective((p_pow - unit_production_cost) * D, GRB.MAXIMIZE)

# 6. Add all constraints.

# Demand Function: D = 100 - 5p
model.addConstr(D == demand_intercept + demand_slope * p, name="Demand_Constraint")

# Power Constraint: p_pow = p^0.9
# Using addGenConstrPow(x, y, a) for y = x^a
model.addGenConstrPow(p, p_pow, price_exponent, name="Power_Calculation")

# 7. Solve the model and print results.
model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"Optimal Price (p): {p.X}")
    print(f"Optimal Demand (D): {D.X}")
    print(f"Maximum Profit: {model.ObjVal}")
    # Output the final answer in the required format
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimization failed or was infeasible.")