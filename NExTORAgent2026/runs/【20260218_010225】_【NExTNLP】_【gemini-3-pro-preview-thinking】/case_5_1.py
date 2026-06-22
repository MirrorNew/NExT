import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
# Parameters List from the prompt
assets = ['A', 'B']
N_assets = 2
r_A = 0.5
r_B = 1.0  # Coefficient for the asset B term (implicit 1.0 * x_B^1.2)
exponent_B = 1.2
lower_bound_A = 1.5
lower_bound_B = 0.0
risk_limit = 9
risk_power = 2

# 2. Create the model.
model = gp.Model("Blue_Ocean_Capital_Optimization")

# 3. Identify function expressions requiring auxiliary variables.
# The objective includes x_B^1.2, which is convex. Maximizing it makes the problem non-convex.
model.Params.NonConvex = 2

# 4. Create decision variables.
# x_A: Amount of capital allocated to asset A
x_A = model.addVar(lb=lower_bound_A, vtype=GRB.CONTINUOUS, name="x_A")

# x_B: Amount of capital allocated to asset B
x_B = model.addVar(lb=lower_bound_B, vtype=GRB.CONTINUOUS, name="x_B")

# Auxiliary variable for the power term substitution: y_B = x_B^1.2
# The values can range widely, set bounds to infinity to rely on constraints
y_B = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y_B")

# 5. Set up the objective function.
# Maximize f(x_A, x_B) = 0.5 * x_A + x_B^1.2
# Represented as r_A * x_A + r_B * y_B
model.setObjective(r_A * x_A + r_B * y_B, GRB.MAXIMIZE)

# 6. Add all constraints.

# Constraint: Risk Tolerance Limit
# x_A^2 + x_B^2 <= 9
model.addConstr(x_A**risk_power + x_B**risk_power <= risk_limit, name="RiskConstraint")

# Constraint: General Constraint for Power Function
# y_B = x_B ^ 1.2
# Note: Pay attention to the order of x and y in addGenConstrPow(x, y, a) -> y = x^a
model.addGenConstrPow(x_B, y_B, exponent_B, name="PowerConstraint_B")

# Note: The constraints x_A >= 1.5 and x_B >= 0 are handled by the 'lb' parameter in addVar.

# 7. Solve the model and print results.
model.optimize()

if model.Status == GRB.OPTIMAL:
    print(f"Optimal Solution Found:")
    print(f"x_A = {x_A.X}")
    print(f"x_B = {x_B.X}")
    print(f"y_B (x_B^{exponent_B}) = {y_B.X}")
    print(f"Objective Value = {model.ObjVal}")
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimization was not successful.")