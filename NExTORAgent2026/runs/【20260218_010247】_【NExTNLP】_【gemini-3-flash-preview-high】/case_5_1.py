import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs
assets = ['A', 'B']
N_assets = 2
r_A = 0.5
r_B = 1.0
exponent_B = 1.2
lower_bound_A = 1.5
lower_bound_B = 0.0
risk_limit = 9
risk_power = 2

# Create the model
model = gp.Model("BlueOceanCapitalOptimization")

# Set the model to NonConvex mode for power functions
model.Params.NonConvex = 2

# 3. Create decision variables
x_A = model.addVar(lb=lower_bound_A, vtype=GRB.CONTINUOUS, name="x_A")
x_B = model.addVar(lb=lower_bound_B, vtype=GRB.CONTINUOUS, name="x_B")

# 4. Create auxiliary substitution variables
# The values of these auxiliary variables should range from negative infinity to positive infinity
y_A_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y_A_sq")
y_B_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y_B_sq")
y_B_pow = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y_B_pow")

# 5. Set up the objective function
# Maximize f(x_A, x_B) = 0.5 * x_A + 1.0 * x_B^1.2
# Using the auxiliary variable y_B_pow for x_B^1.2
model.setObjective(r_A * x_A + r_B * y_B_pow, GRB.MAXIMIZE)

# 6. Add all constraints
# General constraints for powers (y = x^a)
# x_A^2
model.addGenConstrPow(x_A, y_A_sq, float(risk_power))
# x_B^2
model.addGenConstrPow(x_B, y_B_sq, float(risk_power))
# x_B^1.2
model.addGenConstrPow(x_B, y_B_pow, exponent_B)

# Total investment risk constraint: x_A^2 + x_B^2 <= 9
model.addConstr(y_A_sq + y_B_sq <= risk_limit, name="RiskToleranceLimit")

# 7. Solve the model and print results
model.optimize()

if model.status == GRB.OPTIMAL:
    max_return = model.ObjVal
    print(f"FinalAnswer=【{max_return}】")
else:
    print("Optimal solution not found.")