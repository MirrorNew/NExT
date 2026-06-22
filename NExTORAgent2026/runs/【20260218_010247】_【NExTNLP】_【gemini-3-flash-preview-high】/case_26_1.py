import gurobipy as gp
from gurobipy import GRB

# Define all parameter matrices and data inputs
investment_limit = 10
F_linear_coeff = 5
F_quad_coeff = -0.01
S_linear_coeff = 10
startup_consumption = 5
P_threshold = 2
consumption_quad_coeff = 1

# Create the model
model = gp.Model("RenewableEnergyOptimization")

# Create decision variables
W = model.addVar(lb=0, ub=10, vtype=GRB.CONTINUOUS, name="W")
P = model.addVar(lb=0, ub=10, vtype=GRB.CONTINUOUS, name="P")

# Create auxiliary substitution variables as suggested
# Range from negative infinity to positive infinity as per instructions
W_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="W_sq")
P_excess = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_excess")
P_excess_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_excess_sq")
F = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="F")
S = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="S")
C = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="C")

# Binary indicator variable for P > P_threshold
y = model.addVar(vtype=GRB.BINARY, name="y")

# Set up the objective function
# Z = F + S - startup_consumption - C
model.setObjective(F + S - startup_consumption - C, GRB.MAXIMIZE)

# Add all constraints
# 1. Total Investment Limit
model.addConstr(W + P <= investment_limit, name="TotalInvestmentLimit")

# 2. Wind Power Generation Function: F = 5W - 0.01W^2
model.addGenConstrPow(W, W_sq, 2)
model.addConstr(F == F_linear_coeff * W + F_quad_coeff * W_sq, name="WindGeneration")

# 3. Photovoltaic Power Generation Function: S = 10P
model.addConstr(S == S_linear_coeff * P, name="PVGeneration")

# 4. PV Investment Threshold Logic using indicator constraints
# y = 1 if P >= 2, y = 0 if P <= 2
model.addGenConstrIndicator(y, 1, P >= P_threshold)
model.addGenConstrIndicator(y, 0, P <= P_threshold)

# 5. Definition of Excess PV Investment
# If y = 1, P_excess = P - 2; If y = 0, P_excess = 0
model.addGenConstrIndicator(y, 1, P_excess == P - P_threshold)
model.addGenConstrIndicator(y, 0, P_excess == 0)

# 6. Infrastructure Consumption Function: C = P_excess^2
model.addGenConstrPow(P_excess, P_excess_sq, 2)
model.addConstr(C == consumption_quad_coeff * P_excess_sq, name="Consumption")

# Enable non-convex solver for quadratic constraints and general constraints
model.Params.NonConvex = 2

# Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print(f"Wind Investment (W): {W.X} billion yuan")
    print(f"PV Investment (P): {P.X} billion yuan")
    print(f"Total Annual Power Generation (Z): {model.ObjVal} billion kWh")
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimal solution was not found.")