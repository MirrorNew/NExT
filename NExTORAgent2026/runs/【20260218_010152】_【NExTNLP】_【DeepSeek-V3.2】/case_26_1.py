import gurobipy as gp
from gurobipy import GRB

# Parameters List
investment_limit = 10.0
F_linear_coeff = 5.0
F_quad_coeff = -0.01
S_linear_coeff = 10.0
startup_consumption = 5.0
P_threshold = 2.0
consumption_quad_coeff = 1.0

# Create model
model = gp.Model("RenewableEnergyInvestment")

# Create decision variables
W = model.addVar(lb=0.0, ub=10.0, name="W")  # Investment in wind power (billion yuan)
P = model.addVar(lb=0.0, ub=10.0, name="P")  # Investment in photovoltaic power (billion yuan)

# Create auxiliary substitution variables
F = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="F")  # Wind power generation
S = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="S")  # PV power generation before deductions
C = model.addVar(lb=0.0, ub=GRB.INFINITY, name="C")  # Infrastructure consumption
WW = model.addVar(lb=0.0, ub=100.0, name="WW")  # W² auxiliary variable
D = model.addVar(lb=-2.0, ub=8.0, name="D")  # P - 2 (excess over threshold)
y = model.addVar(vtype=GRB.BINARY, name="y")  # Binary indicator for P ≥ 2

# Create auxiliary variable for D²
DD = model.addVar(lb=0.0, ub=64.0, name="DD")  # D²

# Set non-convex parameter for quadratic constraints
model.Params.NonConvex = 2

# Set up the objective function
Z = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Z")
model.setObjective(Z, GRB.MAXIMIZE)

# Add all constraints
# Total investment limit
model.addConstr(W + P <= investment_limit, "total_investment")

# Wind power generation function: F = 5W - 0.01W²
model.addGenConstrPow(W, WW, 2, "WW_eq")
model.addConstr(F == F_linear_coeff * W + F_quad_coeff * WW, "wind_generation")

# PV power generation function: S = 10P
model.addConstr(S == S_linear_coeff * P, "pv_generation")

# Define D = P - 2
model.addConstr(D == P - P_threshold, "excess_definition")

# Indicator constraints for P ≥ 2
model.addGenConstrIndicator(y, 1, P >= P_threshold, "indicator_ge")
model.addGenConstrIndicator(y, 0, P <= P_threshold, "indicator_le")

# Infrastructure consumption: C = (max{P-2,0})² = D² * y
model.addGenConstrPow(D, DD, 2, "DD_eq")
# Create auxiliary variable for D² * y
CC = model.addVar(lb=0.0, ub=64.0, name="CC")  # D² * y
model.addConstr(CC == DD * y, "CC_eq")
model.addConstr(C == consumption_quad_coeff * CC, "infrastructure_consumption")

# Objective definition: Z = F + S - startup_consumption - C
model.addConstr(Z == F + S - startup_consumption - C, "objective_def")

# Solve the model and print results
model.optimize()

if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    print(f"Wind investment W = {W.X:.4f} billion yuan")
    print(f"PV investment P = {P.X:.4f} billion yuan")
    print(f"Wind generation F = {F.X:.4f} billion kWh")
    print(f"PV generation S = {S.X:.4f} billion kWh")
    print(f"Infrastructure consumption C = {C.X:.4f} billion kWh")
    print(f"Total generation Z = {Z.X:.4f} billion kWh")
    
    # Final answer: maximum total power generation
    print(f"FinalAnswer=【{Z.X:.4f}】")
else:
    print(f"No optimal solution found. Status: {model.status}")
    print(f"FinalAnswer=【0】")