import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and any other necessary packages.
# (Already imported above)

# 2. Define all parameter matrices and data inputs.
# Parameters from the list
frequency_bands = [2.6, 3.5, 4.9]
g = [0.5, 1.2, 0.9]  # Gain coefficients for channels 1, 2, 3
P_total_max = 20
num_channels = 3

# 3. Create decision variables.
model = gp.Model("PowerAllocation")

# P[i] represents the power allocated to channel i
# Indices 0, 1, 2 correspond to P1, P2, P3
P = model.addVars(num_channels, lb=0.0, ub=P_total_max, vtype=GRB.CONTINUOUS, name="P")

# 4. Create any auxiliary substitution or indicator variables in coding advice.
# Advice: "Introduce auxiliary variables X_1, X_2, X_3 to capture the linear arguments... X_i = 1 + g_i * P_i"
# Advice: "Introduce auxiliary variables R_1, R_2, R_3 to capture the logarithmic values... R_i = log2(X_i)"
# Note: We will use Y variable names for R (Rate) to match standard generic naming or stick to R as per advice logic.
# Let's use X for arguments and R_vars for rates.
X = model.addVars(num_channels, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="X")
R_vars = model.addVars(num_channels, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="R_val")

# 5. Set up the objective function.
# Maximize R_total = sum(R_vars[i])
model.setObjective(gp.quicksum(R_vars[i] for i in range(num_channels)), GRB.MAXIMIZE)

# 6. Add all constraints (including gen‐constr and indicator constraints).

# Constraint 1: Total power limit
# P1 + P2 + P3 <= 20
model.addConstr(gp.quicksum(P[i] for i in range(num_channels)) <= P_total_max, name="TotalPowerLimit")

# Constraint 2: Auxiliary variable definitions and functional constraints
# Advice: model.addConstr(X[i] == 1 + g[i] * P[i])
for i in range(num_channels):
    model.addConstr(X[i] == 1 + g[i] * P[i], name=f"Def_X_{i}")

# Advice: model.addGenConstrLogA(X[i], R[i], 2.0)
# This enforces R_vars[i] = log2(X[i])
for i in range(num_channels):
    model.addGenConstrLogA(X[i], R_vars[i], 2.0, name=f"Log2_Constraint_{i}")

# Advice: Set model.Params.NonConvex = 2
model.Params.NonConvex = 2

# 7. Solve the model and print results.
model.optimize()

if model.Status == GRB.OPTIMAL:
    print("\nOptimization Successful!")
    print(f"Total Max Rate: {model.ObjVal:.4f}")
    for i in range(num_channels):
        print(f"Channel {i+1} ({frequency_bands[i]}GHz, g={g[i]}): Power = {P[i].X:.4f} W, Rate = {R_vars[i].X:.4f}")
    
    # ATTENTION 1: Output the final answer in the specific format
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("No optimal solution found.")
    print(f"FinalAnswer=【None】")