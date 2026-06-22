import gurobipy as gp

# ==================== Parameters ====================
T = 4
initial_storage = 1000000.0  # cubic meters
S_min = 0.0
S_max = 2000000.0
final_storage_min = 200000.0

# Convert to million cubic meters for numerical stability
scale = 1000000.0
initial_storage = initial_storage / scale
S_min = S_min / scale
S_max = S_max / scale
final_storage_min = final_storage_min / scale

# Inflows (million m³)
I = [80/scale, 50/scale, 20/scale, 0/scale]

# Maximum supply (million m³)
MaxA = [20/scale, 40/scale, 55/scale, 50/scale]
MaxB = [10/scale, 30/scale, 40/scale, 40/scale]

# Minimum supply (million m³)
MinA = [10/scale, 20/scale, 30/scale, 20/scale]
MinB = [8/scale, 20/scale, 30/scale, 34/scale]

# Power generation multipliers
gen_multiplier = [2, 2, 1, 1]

# Small epsilon for indicator constraints
eps = 1e-6

# ==================== Model ====================
model = gp.Model("Chenxi_Reservoir_Optimization")

# ==================== Variables ====================
# Storage at end of period t (S0, S1, S2, S3, S4)
S = {}
for t in range(T+1):
    S[t] = model.addVar(lb=S_min, ub=S_max, name=f"S_{t}")

# Total water release in period t
R = {}
for t in range(1, T+1):
    R[t] = model.addVar(lb=0.0, ub=1.0, name=f"R_{t}")

# Water supplied to region A and B
R_A = {}
R_B = {}
for t in range(1, T+1):
    R_A[t] = model.addVar(lb=0.0, ub=MaxA[t-1], name=f"R_A_{t}")
    R_B[t] = model.addVar(lb=0.0, ub=MaxB[t-1], name=f"R_B_{t}")

# Power generation in period t
P = {}
for t in range(1, T+1):
    P[t] = model.addVar(lb=0.0, name=f"P_{t}")

# Binary variables for minimum demand satisfaction
y_A = {}
y_B = {}
for t in range(1, T+1):
    y_A[t] = model.addVar(vtype=gp.GRB.BINARY, name=f"y_A_{t}")
    y_B[t] = model.addVar(vtype=gp.GRB.BINARY, name=f"y_B_{t}")

# ==================== Objective ====================
model.setObjective(gp.quicksum(P[t] for t in range(1, T+1)), sense=gp.GRB.MAXIMIZE)

# ==================== Constraints ====================
# Initial storage
model.addConstr(S[0] == initial_storage, name="initial_storage")

# Water balance
for t in range(1, T+1):
    model.addConstr(S[t] == S[t-1] + I[t-1] - R[t], name=f"water_balance_{t}")

# Release cannot exceed available water
for t in range(1, T+1):
    model.addConstr(R[t] <= S[t-1] + I[t-1], name=f"release_availability_{t}")

# Supply equality: R_A + R_B = R
for t in range(1, T+1):
    model.addConstr(R_A[t] + R_B[t] == R[t], name=f"supply_equality_{t}")

# Power generation relation
for t in range(1, T+1):
    model.addConstr(P[t] == gen_multiplier[t-1] * R[t], name=f"power_relation_{t}")

# Minimum demand constraints using indicator constraints
for t in range(1, T+1):
    # Region A
    # If y_A[t] = 1, then R_A[t] >= MinA[t-1]
    model.addGenConstrIndicator(y_A[t], 1, R_A[t] >= MinA[t-1], name=f"indicator_min_A_1_{t}")
    # If y_A[t] = 0, then R_A[t] <= MinA[t-1] - eps
    model.addGenConstrIndicator(y_A[t], 0, R_A[t] <= MinA[t-1] - eps, name=f"indicator_min_A_0_{t}")
    
    # Region B
    # If y_B[t] = 1, then R_B[t] >= MinB[t-1]
    model.addGenConstrIndicator(y_B[t], 1, R_B[t] >= MinB[t-1], name=f"indicator_min_B_1_{t}")
    # If y_B[t] = 0, then R_B[t] <= MinB[t-1] - eps
    model.addGenConstrIndicator(y_B[t], 0, R_B[t] <= MinB[t-1] - eps, name=f"indicator_min_B_0_{t}")

# Final storage requirement
model.addConstr(S[T] >= final_storage_min, name="final_storage")

# ==================== Solve ====================
model.optimize()

# ==================== Results ====================
if model.status == gp.GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"Total power generation: {model.ObjVal:.6f} million units")
    
    print("\nDetailed schedule:")
    for t in range(1, T+1):
        print(f"Period {t}:")
        print(f"  Storage at start: {S[t-1].X:.6f} million m³")
        print(f"  Inflow: {I[t-1]:.6f} million m³")
        print(f"  Release total: {R[t].X:.6f} million m³")
        print(f"  Release to A: {R_A[t].X:.6f} million m³ (min: {MinA[t-1]:.6f}, max: {MaxA[t-1]:.6f})")
        print(f"  Release to B: {R_B[t].X:.6f} million m³ (min: {MinB[t-1]:.6f}, max: {MaxB[t-1]:.6f})")
        print(f"  Power generation: {P[t].X:.6f} million units (multiplier: {gen_multiplier[t-1]})")
        print(f"  Storage at end: {S[t].X:.6f} million m³")
        print()
    
    print(f"Final storage S_4: {S[T].X:.6f} million m³ (≥ {final_storage_min:.6f})")
    
    total_power = sum(P[t].X for t in range(1, T+1))
    print(f"\nFinal total power generation: {total_power:.6f} million units")
    print(f"FinalAnswer=【{total_power:.6f}】")
else:
    print("No optimal solution found.")
    if model.status == gp.GRB.INFEASIBLE:
        print("Model is infeasible.")
    elif model.status == gp.GRB.UNBOUNDED:
        print("Model is unbounded.")
    print(f"FinalAnswer=【0】")