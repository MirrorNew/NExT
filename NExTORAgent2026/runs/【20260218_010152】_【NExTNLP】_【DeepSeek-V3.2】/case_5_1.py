import gurobipy as gp
from gurobipy import GRB

# Define parameters from the Parameters List
r_A = 0.5
r_B = 1.0
exponent_B = 1.2
lower_bound_A = 1.5
lower_bound_B = 0.0
risk_limit = 9
risk_power = 2

# Create model
model = gp.Model("AssetAllocation")

# Create decision variables
x_A = model.addVar(lb=lower_bound_A, ub=GRB.INFINITY, name="x_A")
x_B = model.addVar(lb=lower_bound_B, ub=GRB.INFINITY, name="x_B")

# Create auxiliary variables
y_risk_A = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y_risk_A")
y_risk_B = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y_risk_B")
y_return_B = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y_return_B")

# Set model parameters for non-convex optimization
model.Params.NonConvex = 2

# Add constraints linking auxiliary variables to decision variables
model.addGenConstrPow(x_A, y_risk_A, 2, "pow_constr_risk_A")
model.addGenConstrPow(x_B, y_risk_B, 2, "pow_constr_risk_B")
model.addGenConstrPow(x_B, y_return_B, exponent_B, "pow_constr_return_B")

# Add risk tolerance constraint
model.addConstr(y_risk_A + y_risk_B <= risk_limit, "RiskToleranceLimit")

# Set objective function
objective = r_A * x_A + y_return_B
model.setObjective(objective, GRB.MAXIMIZE)

# Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print(f"Optimal solution found")
    print(f"Capital allocated to asset A (x_A): {x_A.x}")
    print(f"Capital allocated to asset B (x_B): {x_B.x}")
    print(f"Risk contribution of asset A (x_A^2): {y_risk_A.x}")
    print(f"Risk contribution of asset B (x_B^2): {y_risk_B.x}")
    print(f"Return from asset B (x_B^{exponent_B}): {y_return_B.x}")
    print(f"Maximum return: {model.objVal}")
    
    # Output the answer to the question
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print(f"No optimal solution found. Status: {model.status}")
    print(f"FinalAnswer=【No optimal solution】")