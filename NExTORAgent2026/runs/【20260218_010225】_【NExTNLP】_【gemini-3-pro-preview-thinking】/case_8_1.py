import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and define all parameter matrices and data inputs
# Parameters from the list
price_A = [70.0, -4.0]  # [Intercept, Slope]
price_B = [150.0, -15.0] # [Intercept, Slope]
unit_production_cost = 150000 # Yuan
machine_loss_fee = 1000000    # Yuan

# Data Processing: Convert monetary values to '10,000 yuan' units to match the price function units
# The price functions are in units of 10,000 yuan.
# Cost 150,000 yuan = 15 * 10,000 yuan
# Fixed cost 1,000,000 yuan = 100 * 10,000 yuan
unit_cost_scaled = unit_production_cost / 10000.0
fixed_cost_scaled = machine_loss_fee / 10000.0

# 2. Create model and set NonConvex parameter
model = gp.Model("ProfitMaximization")
model.Params.NonConvex = 2

# 3. Create decision variables
# Demand quantities for Customer A and Customer B
x1 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="x1")
x2 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="x2")

# 4. Create auxiliary substitution variables
# For quadratic terms in the objective function: x1^2 and x2^2
aux_sq_x1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="aux_sq_x1")
aux_sq_x2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="aux_sq_x2")

# 5. Add constraints
# General constraints for power functions: aux_sq = x^2
# Order: (xvar, yvar, a) => yvar = xvar^a
model.addGenConstrPow(x1, aux_sq_x1, 2.0, "pow_x1")
model.addGenConstrPow(x2, aux_sq_x2, 2.0, "pow_x2")

# Implicit constraints: Prices must be non-negative for the demand model to make sense
# p1 = 70 - 4x1 >= 0
model.addConstr(price_A[0] + price_A[1] * x1 >= 0, "PriceA_NonNeg")
# p2 = 150 - 15x2 >= 0
model.addConstr(price_B[0] + price_B[1] * x2 >= 0, "PriceB_NonNeg")

# 6. Set up the objective function
# Maximize Profit Z
# Revenue A = x1 * (70 - 4x1) = 70x1 - 4x1^2
# Revenue B = x2 * (150 - 15x2) = 150x2 - 15x2^2
# Total Cost = unit_cost_scaled * (x1 + x2) + fixed_cost_scaled
# Profit = (70 * x1 - 4 * aux_sq_x1) + (150 * x2 - 15 * aux_sq_x2) - 15 * (x1 + x2) - 100
# Profit = (70 - 15) * x1 - 4 * aux_sq_x1 + (150 - 15) * x2 - 15 * aux_sq_x2 - 100

coeff_linear_x1 = price_A[0] - unit_cost_scaled
coeff_quad_x1 = price_A[1]  # -4.0
coeff_linear_x2 = price_B[0] - unit_cost_scaled
coeff_quad_x2 = price_B[1]  # -15.0

objective_expr = (
    coeff_linear_x1 * x1 + 
    coeff_quad_x1 * aux_sq_x1 + 
    coeff_linear_x2 * x2 + 
    coeff_quad_x2 * aux_sq_x2 - 
    fixed_cost_scaled
)

model.setObjective(objective_expr, GRB.MAXIMIZE)

# 7. Solve the model and print results
model.optimize()

if model.Status == GRB.OPTIMAL:
    print(f"Optimal x1: {x1.X}")
    print(f"Optimal x2: {x2.X}")
    print(f"Maximum Profit: {model.ObjVal}")
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimization failed.")