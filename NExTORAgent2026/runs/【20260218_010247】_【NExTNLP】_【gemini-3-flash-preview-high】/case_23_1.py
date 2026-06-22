import gurobipy as gp
from gurobipy import GRB

# Define parameter matrices and data inputs
frequency_bands_count = 3
frequency_bands = [2.6, 3.5, 4.9]
channels_count = 3
channels = [2.6, 3.5, 4.9]
P_total_max = 20
g = [0.5, 1.2, 0.9]
P_i_min = 0

# Create the model
model = gp.Model("5G_Power_Allocation")

# Create decision variables
# P_i represents the power allocated to channel i
P = {}
for i in range(channels_count):
    P[i] = model.addVar(lb=P_i_min, ub=P_total_max, vtype=GRB.CONTINUOUS, name=f"P_{i}")

# Create auxiliary substitution variables
# X_i will store (1 + g_i * P_i)
# R_i will store log2(X_i)
X = {}
R = {}
for i in range(channels_count):
    # As per instructions, auxiliary variables have lb=-GRB.INFINITY and ub=GRB.INFINITY
    X[i] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"X_{i}")
    R[i] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"R_{i}")

# Set up the objective function: Maximize R_total = sum(R_i)
model.setObjective(gp.quicksum(R[i] for i in range(channels_count)), GRB.MAXIMIZE)

# Set NonConvex parameter for general constraints (Log)
model.Params.NonConvex = 2

# Add constraints
# 1. Total transmission power limit
model.addConstr(gp.quicksum(P[i] for i in range(channels_count)) <= P_total_max, "TotalPowerLimit")

# 2. Map power to auxiliary log arguments and log values
for i in range(channels_count):
    # Definition of the log argument: X_i = 1 + g_i * P_i
    model.addConstr(X[i] == 1 + g[i] * P[i], name=f"Argument_Constraint_{i}")
    
    # Definition of the channel rate: R_i = log2(X_i)
    # Gurobi's addGenConstrLogA(x, y, base) represents y = log_base(x)
    model.addGenConstrLogA(X[i], R[i], 2.0, name=f"Log_Constraint_{i}")

# Solve the model
model.optimize()

# Print results and the final answer
if model.status == GRB.OPTIMAL:
    print(f"Optimal Total Rate: {model.objVal}")
    print(f"FinalAnswer=【{model.objVal}】")
else:
    # If the solver reached an alternative termination status, try to print the best objective value found
    try:
        print(f"FinalAnswer=【{model.objVal}】")
    except AttributeError:
        print("FinalAnswer=【Optimal solution not found】")