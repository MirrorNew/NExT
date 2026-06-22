import gurobipy as gp
from gurobipy import GRB

# Define parameters from the provided list
frequency_bands_count = 3
frequency_bands = [2.6, 3.5, 4.9]
channels_count = 3
channels = [2.6, 3.5, 4.9]
P_total_max = 20
g = [0.5, 1.2, 0.9]
P_i_min = 0

# Create model
model = gp.Model("5G_Power_Allocation")

# Create decision variables for power allocation
P = model.addVars(channels_count, lb=P_i_min, ub=P_total_max, name="P")

# Create auxiliary variables Z_i = 1 + g_i * P_i
# Fix the error: compute upper bounds correctly using list comprehension
Z_ub = [1 + g[i] * P_total_max for i in range(channels_count)]
Z = model.addVars(channels_count, lb=1, ub=Z_ub, name="Z")

# Create auxiliary variables R_i = log2(Z_i)
R = model.addVars(channels_count, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="R")

# Set non-convex parameter for logarithmic constraints
model.Params.NonConvex = 2

# Add constraints: Z_i = 1 + g_i * P_i
for i in range(channels_count):
    model.addConstr(Z[i] == 1 + g[i] * P[i], name=f"Z_constraint_{i}")

# Add logarithmic constraints: R_i = log2(Z_i)
for i in range(channels_count):
    model.addGenConstrLogA(Z[i], R[i], 2, name=f"log_constraint_{i}")

# Add total power constraint
model.addConstr(gp.quicksum(P[i] for i in range(channels_count)) <= P_total_max, name="total_power")

# Set objective: maximize total rate = sum of R_i
model.setObjective(gp.quicksum(R[i] for i in range(channels_count)), GRB.MAXIMIZE)

# Optimize the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"Objective value (total rate): {model.ObjVal:.4f} bits/s/Hz")
    
    for i in range(channels_count):
        print(f"P_{i+1} (Power for channel {channels[i]} GHz): {P[i].X:.4f} W")
        print(f"  Z_{i+1} = 1 + g_{i+1}*P_{i+1}: {Z[i].X:.4f}")
        print(f"  R_{i+1} = log2(Z_{i+1}): {R[i].X:.4f} bits/s/Hz")
    
    total_power = sum(P[i].X for i in range(channels_count))
    print(f"\nTotal power used: {total_power:.4f} W (max allowed: {P_total_max} W)")
    
    # Extract the final answer - the objective value (total rate)
    final_answer = model.ObjVal
else:
    print(f"Optimization failed with status: {model.status}")
    final_answer = None

print(f"FinalAnswer=【{final_answer}】")