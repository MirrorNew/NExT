import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters and data
# =========================

# From Parameters List
T = 4
initial_storage = 1000000.0
natural_loss = 0.0
regions = ['A', 'B']
x_max = 1000000.0
x_min = 0.0
S_min = 0.0
S_max = 2000000.0
gen_multiplier = [2, 2, 1, 1]  # for t = 1..4
final_storage_min = 200000.0
Table_1_Period_Data = {
    'Period': [1, 2, 3, 4],
    'Inflow': [80, 50, 20, 0],
    'MaxSupplyA': [20, 40, 55, 50],
    'MaxSupplyB': [10, 30, 40, 40],
    'MinSupplyA': [10, 20, 30, 20],
    'MinSupplyB': [8, 20, 30, 34],
    'GenRate': [0.31, 1.55, 2.05, 0.65]
}

# Convert tabular data to dictionaries indexed by period (1..T)
periods = Table_1_Period_Data['Period']

Inflow = {t: Table_1_Period_Data['Inflow'][t - 1] * 10000.0 for t in periods}
# Note: original table uses "80, 50, 20, 0" versus reservoir 1,000,000.
# We'll consistently treat them as 10,000 m^3 units: 80 -> 800,000, etc.

MaxSupplyA = {t: Table_1_Period_Data['MaxSupplyA'][t - 1] * 10000.0 for t in periods}
MaxSupplyB = {t: Table_1_Period_Data['MaxSupplyB'][t - 1] * 10000.0 for t in periods}
MinSupplyA = {t: Table_1_Period_Data['MinSupplyA'][t - 1] * 10000.0 for t in periods}
MinSupplyB = {t: Table_1_Period_Data['MinSupplyB'][t - 1] * 10000.0 for t in periods}

# Release and storage bounds
R_min = x_min          # 0
R_max = x_max          # 1,000,000 per period
S_lower = S_min        # 0
S_upper = S_max        # 2,000,000

# =========================
# 2. Create model
# =========================

model = gp.Model("Chenxi_Reservoir_Scheduling")

# =========================
# 3. Decision variables
# =========================

# Total release in each period
R = model.addVars(periods, lb=R_min, ub=R_max, vtype=GRB.CONTINUOUS, name="R")

# Regional supplies
RA = model.addVars(periods, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="RA")
RB = model.addVars(periods, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="RB")

# Storage at end of each period
S = model.addVars(periods, lb=S_lower, ub=S_upper, vtype=GRB.CONTINUOUS, name="S")

# Power generation in each period
P = model.addVars(periods, lb=0.0, vtype=GRB.CONTINUOUS, name="P")

# Auxiliary: initial storage S_0
S0 = model.addVar(lb=S_lower, ub=S_upper, vtype=GRB.CONTINUOUS, name="S0")

# =========================
# 4. Constraints
# =========================

# Initial storage
model.addConstr(S0 == initial_storage, name="Initial_Storage")

# Water balance and release availability
for t in periods:
    if t == 1:
        prev_S = S0
    else:
        prev_S = S[t - 1]

    # Water balance: S_t = S_{t-1} + Inflow_t - R_t - natural_loss
    model.addConstr(
        S[t] == prev_S + Inflow[t] - R[t] - natural_loss,
        name=f"Water_Balance_t{t}"
    )

    # Release cannot exceed available water
    model.addConstr(
        R[t] <= prev_S + Inflow[t],
        name=f"Release_Availability_t{t}"
    )

    # Total release bounds
    model.addConstr(
        R[t] >= R_min,
        name=f"Release_LB_t{t}"
    )
    model.addConstr(
        R[t] <= R_max,
        name=f"Release_UB_t{t}"
    )

    # Storage bounds (already in variable definition but added explicitly)
    model.addConstr(
        S[t] >= S_lower,
        name=f"Storage_LB_t{t}"
    )
    model.addConstr(
        S[t] <= S_upper,
        name=f"Storage_UB_t{t}"
    )

# Final storage minimum requirement
model.addConstr(S[4] >= final_storage_min, name="Final_Storage_Min")

# Regional supply linkage and bounds
for t in periods:
    # Link RA, RB to total release R
    model.addConstr(RA[t] + RB[t] == R[t], name=f"Supply_Equality_t{t}")

    # Demand upper bounds
    model.addConstr(RA[t] <= MaxSupplyA[t], name=f"MaxSupplyA_t{t}")
    model.addConstr(RB[t] <= MaxSupplyB[t], name=f"MaxSupplyB_t{t}")

    # Demand lower bounds (ensure downstream demand not affected)
    model.addConstr(RA[t] >= MinSupplyA[t], name=f"MinSupplyA_t{t}")
    model.addConstr(RB[t] >= MinSupplyB[t], name=f"MinSupplyB_t{t}")

# Power-reward relation: P_t = gen_multiplier[t-1] * R_t
for t in periods:
    model.addConstr(
        P[t] == gen_multiplier[t - 1] * R[t],
        name=f"Power_Relation_t{t}"
    )

# =========================
# 5. Objective function
# =========================

# Maximize total power generation over all periods
model.setObjective(gp.quicksum(P[t] for t in periods), GRB.MAXIMIZE)

# =========================
# 6. Solve model
# =========================

model.optimize()

# =========================
# 7. Print results
# =========================

if model.Status == GRB.OPTIMAL:
    total_power = model.ObjVal
    print("Optimal solution found.")
    print(f"Total power generation (objective): {total_power:.4f}")

    print("\nPeriod-wise results:")
    for t in periods:
        print(
            f"t={t}: R={R[t].X:.2f}, RA={RA[t].X:.2f}, RB={RB[t].X:.2f}, "
            f"S={S[t].X:.2f}, P={P[t].X:.2f}"
        )

    print(f"\nFinal storage S4: {S[4].X:.2f}")
else:
    total_power = float('nan')
    print(f"Optimization ended with status {model.Status}, no optimal solution.")

# =========================
# 8. Final answer output
# =========================

# The question asks for the calculated total power generation.
print(f"FinalAnswer=【{total_power}】")