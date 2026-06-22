import gurobipy as gp
from gurobipy import GRB
import math

# ==============================
# 1. Define parameters (from Parameters List)
# ==============================
frequency_bands_count = 3
frequency_bands = [2.6, 3.5, 4.9]
channels_count = 3
channels = [2.6, 3.5, 4.9]
P_total_max = 20
g = [0.5, 1.2, 0.9]
P_i_min = 0

# Index set for channels
I = range(channels_count)

# ==============================
# 2. Create model
# ==============================
model = gp.Model("5G_Power_Allocation_WaterFilling")

# Enable general nonlinear (log) constraints
model.Params.NonConvex = 2

# ==============================
# 3. Decision variables P_i (power allocation)
# ==============================
# P_i >= 0, continuous
P = model.addVars(I, lb=P_i_min, name="P")

# ==============================
# 4. Auxiliary variables
# ==============================
# Z_i = 1 + g_i * P_i  (argument of the log)
Z = model.addVars(I, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Z")

# R_i = log2(Z_i)  (channel rate)
R = model.addVars(I, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="R")

# ==============================
# 5. Objective: maximize total rate sum_i R_i
# ==============================
model.setObjective(gp.quicksum(R[i] for i in I), GRB.MAXIMIZE)

# ==============================
# 6. Constraints
# ==============================

# 6.1 Total power limit: sum_i P_i <= P_total_max
model.addConstr(gp.quicksum(P[i] for i in I) <= P_total_max, name="TotalPowerLimit")

# 6.2 Non-negativity already enforced via lb, but we can add explicit constraints if desired
for i in I:
    model.addConstr(P[i] >= P_i_min, name=f"NonNeg_P_{i+1}")

# 6.3 Define Z_i = 1 + g_i * P_i (linear constraints)
for i in I:
    model.addConstr(Z[i] == 1 + g[i] * P[i], name=f"Z_def_{i+1}")

# 6.4 Define R_i = log2(Z_i) using general log constraints with base 2
for i in I:
    # Gurobi general constraint: R_i = log_2(Z_i)
    model.addGenConstrLogA(Z[i], R[i], 2.0, name=f"log2_def_{i+1}")

# ==============================
# 7. Solve model
# ==============================
model.optimize()

# ==============================
# 8. Print results
# ==============================
if model.status == GRB.OPTIMAL or model.status == GRB.LOCAL_OPTIMAL:
    print("Optimal solution found.")
    for i in I:
        print(f"P_{i+1} (Power on channel {channels[i]} GHz) = {P[i].X:.6f} W")
    R_total = sum(R[i].X for i in I)
    print(f"Total rate R_total = {R_total:.6f} (bits/s/Hz)")

    # Final answer is the value of the objective: total data rate
    print(f"FinalAnswer=【{R_total:.6f}】")
else:
    print(f"Optimization ended with status {model.status}.")
    # If no optimal solution, still provide a fallback final answer
    R_total = float('nan')
    print(f"FinalAnswer=【{R_total}】")