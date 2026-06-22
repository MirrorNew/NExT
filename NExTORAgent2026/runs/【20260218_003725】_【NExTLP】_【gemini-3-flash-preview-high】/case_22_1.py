import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
# From Parameters List:
T = 4
initial_storage = 1000000.0
S_min = 0.0
S_max = 2000000.0
x_min = 0.0
x_max = 1000000.0
final_storage_min = 200000.0
gen_multiplier = [2, 2, 1, 1]

# From Table 1 Period Data:
inflow = [80, 50, 20, 0]
max_A = [20, 40, 55, 50]
max_B = [10, 30, 40, 40]
min_A = [10, 20, 30, 20]
min_B = [8, 20, 30, 34]

# 2. Create the optimization model.
model = gp.Model("Chenxi_Reservoir_Scheduling")

# 3. Create decision variables.
# R_t: total water release in period t
R = model.addVars(range(1, T + 1), lb=x_min, ub=x_max, vtype=GRB.CONTINUOUS, name="R")
# RA_t: water supplied to region A in period t
RA = model.addVars(range(1, T + 1), lb=0, vtype=GRB.CONTINUOUS, name="RA")
# RB_t: water supplied to region B in period t
RB = model.addVars(range(1, T + 1), lb=0, vtype=GRB.CONTINUOUS, name="RB")
# S_t: storage at end of period t (S[0] is initial storage)
S = model.addVars(range(0, T + 1), lb=S_min, ub=S_max, vtype=GRB.CONTINUOUS, name="S")
# P_t: power generation in period t
P = model.addVars(range(1, T + 1), lb=0, vtype=GRB.CONTINUOUS, name="P")

# 5. Set up the objective function.
# Maximize total power generation over the four periods.
model.setObjective(gp.quicksum(P[t] for t in range(1, T + 1)), GRB.MAXIMIZE)

# 6. Add all constraints.
# Initial_Storage constraint
model.addConstr(S[0] == initial_storage, name="Initial_Storage")

for t in range(1, T + 1):
    # Water_Balance: Storage at end of period t is storage at start + inflow - release
    # I_t and Max/Min supplies are indexed by t-1 due to 0-indexing in Python lists
    model.addConstr(S[t] == S[t-1] + inflow[t-1] - R[t], name=f"Water_Balance_{t}")

    # Supply_Equality: Total release is the sum of supply to area A and B
    model.addConstr(R[t] == RA[t] + RB[t], name=f"Supply_Equality_{t}")

    # Demand Limits for Region A: must meet minimum and not exceed maximum
    model.addConstr(RA[t] >= min_A[t-1], name=f"Min_Supply_A_{t}")
    model.addConstr(RA[t] <= max_A[t-1], name=f"Max_Supply_A_{t}")

    # Demand Limits for Region B: must meet minimum and not exceed maximum
    model.addConstr(RB[t] >= min_B[t-1], name=f"Min_Supply_B_{t}")
    model.addConstr(RB[t] <= max_B[t-1], name=f"Max_Supply_B_{t}")

    # Power_Generation definition based on the reward mechanism
    model.addConstr(P[t] == gen_multiplier[t-1] * R[t], name=f"Power_Generation_{t}")

# Final_Storage: After 4 periods, inventory must be at least final_storage_min
model.addConstr(S[T] >= final_storage_min, name="Final_Storage_Limit")

# 7. Solve the model and print results.
model.optimize()

if model.status == GRB.OPTIMAL:
    # Print the objective value which is the total power generation.
    print(f"FinalAnswer=【{model.objVal}】")