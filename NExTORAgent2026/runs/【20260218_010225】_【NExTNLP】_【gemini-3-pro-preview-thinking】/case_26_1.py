import gurobipy as gp
from gurobipy import GRB

# Create model
model = gp.Model("EnergyOptimization")

# Define parameters based on the Parameters List
investment_limit = 10.0
F_linear_coeff = 5.0
F_quad_coeff = -0.01
S_linear_coeff = 10.0
startup_consumption = 5.0
P_threshold = 2.0
consumption_quad_coeff = 1.0

# Set parameter to handle non-convex quadratic constraints/objectives
model.Params.NonConvex = 2

# Create decision variables
W = model.addVar(lb=0.0, ub=investment_limit, name="W")
P = model.addVar(lb=0.0, ub=investment_limit, name="P")

# Create auxiliary variables for substitution
# W_sq represents W^2
W_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="W_sq")
# P_excess represents max(P - P_threshold, 0)
P_excess = model.addVar(lb=0.0, ub=GRB.INFINITY, name="P_excess")
# C_val represents P_excess^2
C_val = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="C_val")

# Set up the objective function
# Objective: Maximize Z = F(W) + S(P) - Startup - C(P)
# F(W) = 5W - 0.01W^2
# S(P) = 10P
# C(P) = (max(P-2, 0))^2 => C_val
# Z = 5W - 0.01*W_sq + 10P - 5 - 1*C_val

objective_expr = (F_linear_coeff * W) + (F_quad_coeff * W_sq) + \
                 (S_linear_coeff * P) - startup_consumption - \
                 (consumption_quad_coeff * C_val)

model.setObjective(objective_expr, GRB.MAXIMIZE)

# Add constraints
# 1. Total Investment Limit
model.addConstr(W + P <= investment_limit, name="Total_Investment_Limit")

# 2. Logic for P_excess
# P_excess must be >= P - 2. Since the objective effectively minimizes C_val (and thus P_excess),
# and P_excess has a lower bound of 0, P_excess will equal max(P - 2, 0).
model.addConstr(P_excess >= P - P_threshold, name="P_excess_constraint")

# 3. Quadratic Substitutions using General Constraints
# W^2 = W_sq
model.addGenConstrPow(W, W_sq, 2, name="GenConstr_W_sq")
# P_excess^2 = C_val
model.addGenConstrPow(P_excess, C_val, 2, name="GenConstr_C_val")

# Solve the model
model.optimize()

# Print results
if model.Status == GRB.OPTIMAL:
    print(f"Optimal Objective Value: {model.ObjVal}")
    print(f"Optimal W: {W.X}")
    print(f"Optimal P: {P.X}")
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Model did not solve to optimality.")