import gurobipy as gp
from gurobipy import GRB

# Define parameters
saturation_threshold_TV = 600000
expansion_factor_weak_channel = 3
total_budget_initial = 100
max_investment = {'A': 60, 'B': 50, 'C': 50}
revenue_coefficients = {'A': 30.23, 'B': 24.36, 'C': 20.12}
extra_investment_factor = 3
max_total_budget = 100

# Create model
model = gp.Model("RhodeIslandMedia_MMM_Optimization")

# Enable Gurobi to solve non-convex problems (required for power and indicator constraints)
model.Params.NonConvex = 2

# Decision variables
x_A = model.addVar(lb=0, ub=max_investment['A'], name="x_A")
x_B = model.addVar(lb=0, ub=max_investment['B'], name="x_B")
x_C = model.addVar(lb=0, ub=max_investment['C'], name="x_C")

# Binary indicators for lowest investment channel
delta_A = model.addVar(vtype=GRB.BINARY, name="delta_A")
delta_B = model.addVar(vtype=GRB.BINARY, name="delta_B")
delta_C = model.addVar(vtype=GRB.BINARY, name="delta_C")

# Auxiliary substitution variables (lb=-GRB.INFINITY as requested)
E_A = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="E_A")
E_B = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="E_B")
E_C = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="E_C")

S_A = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="S_A")
S_B = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="S_B")
S_C = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="S_C")

R_A = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="R_A")
R_B = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="R_B")
R_C = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="R_C")

y_A = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y_A")
y_B = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y_B")
y_C = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y_C")

# Set up the objective function
model.setObjective(y_A + y_B + y_C, GRB.MAXIMIZE)

# Constraints

# C1. TV saturation threshold (using explicit limit from problem description)
model.addConstr(x_A <= saturation_threshold_TV)

# C2 & C11. Budget constraints (Total investment including secondary cannot exceed 100)
model.addConstr(x_A + x_B + x_C + E_A + E_B + E_C <= max_total_budget)
model.addConstr(x_A + x_B + x_C <= total_budget_initial)

# C6. Exactly one channel must be the lowest input channel for amplifier logic
model.addConstr(delta_A + delta_B + delta_C == 1)

# C7-C9. Lowest identification using indicator constraints
model.addGenConstrIndicator(delta_A, 1, x_A <= x_B)
model.addGenConstrIndicator(delta_A, 1, x_A <= x_C)
model.addGenConstrIndicator(delta_B, 1, x_B <= x_A)
model.addGenConstrIndicator(delta_B, 1, x_B <= x_C)
model.addGenConstrIndicator(delta_C, 1, x_C <= x_A)
model.addGenConstrIndicator(delta_C, 1, x_C <= x_B)

# C10. Secondary investment logic using indicator constraints
model.addGenConstrIndicator(delta_A, 1, E_A == extra_investment_factor * x_A)
model.addGenConstrIndicator(delta_A, 0, E_A == 0)
model.addGenConstrIndicator(delta_B, 1, E_B == extra_investment_factor * x_B)
model.addGenConstrIndicator(delta_B, 0, E_B == 0)
model.addGenConstrIndicator(delta_C, 1, E_C == extra_investment_factor * x_C)
model.addGenConstrIndicator(delta_C, 0, E_C == 0)

# Substitution for total investment in each channel
model.addConstr(S_A == x_A + E_A)
model.addConstr(S_B == x_B + E_B)
model.addConstr(S_C == x_C + E_C)

# C12. Revenue function definitions using power constraints for square root
model.addGenConstrPow(S_A, R_A, 0.5)
model.addGenConstrPow(S_B, R_B, 0.5)
model.addGenConstrPow(S_C, R_C, 0.5)

model.addConstr(y_A == revenue_coefficients['A'] * R_A)
model.addConstr(y_B == revenue_coefficients['B'] * R_B)
model.addConstr(y_C == revenue_coefficients['C'] * R_C)

# Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimization was not successful.")