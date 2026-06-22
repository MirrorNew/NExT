import gurobipy as gp
from gurobipy import GRB

# ==============================
# 1. Define all parameter values
# ==============================

demand_intercept     = 100   # from Parameters List
demand_slope         = -5    # from Parameters List
unit_production_cost = 10    # from Parameters List
price_exponent       = 0.9   # from Parameters List
price_lower_bound    = 0     # from Parameters List
price_upper_bound    = 20    # from Parameters List

# ==============================
# 2. Create model
# ==============================

model = gp.Model("Drug_Pricing_Optimization")

# Allow nonconvex features (power, bilinear terms)
model.Params.NonConvex = 2

# ==============================
# 3. Decision variables
# ==============================

# p: unit price of the drug (yuan)
p = model.addVar(lb=price_lower_bound, ub=price_upper_bound, vtype=GRB.CONTINUOUS, name="p")

# D: market demand (units), range [0, 100]
D = model.addVar(lb=0.0, ub=demand_intercept, vtype=GRB.CONTINUOUS, name="D")

# ==============================
# 4. Auxiliary variables
# ==============================

# t: t = p^0.9
t = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="t")

# Pi: profit = (t - 10) * D
Pi = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Pi")

# ==============================
# 5. Constraints
# ==============================

# Demand function: D = demand_intercept + demand_slope * p
model.addConstr(D == demand_intercept + demand_slope * p, name="Demand_Function")

# Power relation: t = p^0.9
model.addGenConstrPow(p, t, price_exponent, name="Price_Power")

# Profit definition: Pi = (t - unit_production_cost) * D
model.addConstr(Pi == (t - unit_production_cost) * D, name="Profit_Definition")

# Explicit price bounds (already in variable, but added as constraints per spec)
model.addConstr(p >= price_lower_bound, name="Price_LB")
model.addConstr(p <= price_upper_bound, name="Price_UB")

# ==============================
# 6. Objective: maximize profit
# ==============================

model.setObjective(Pi, GRB.MAXIMIZE)

# ==============================
# 7. Solve model
# ==============================

model.optimize()

# ==============================
# 8. Print results
# ==============================

if model.status == GRB.OPTIMAL:
    optimal_p  = p.X
    optimal_D  = D.X
    optimal_Pi = Pi.X

    print(f"Optimal price p: {optimal_p:.6f}")
    print(f"Optimal demand D: {optimal_D:.6f}")
    print(f"Maximum profit Pi: {optimal_Pi:.6f}")
else:
    print("No optimal solution found.")
    optimal_Pi = float('nan')

# Final answer is the maximum profit
print(f"FinalAnswer=【{optimal_Pi}】")