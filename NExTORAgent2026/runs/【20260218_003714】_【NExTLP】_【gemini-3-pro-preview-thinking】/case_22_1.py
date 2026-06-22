import gurobipy as gp
from gurobipy import GRB

# Create the model
model = gp.Model("Chenxi_Reservoir_Scheduling")

# Parameters List
T = 4
initial_storage = 1000000.0
natural_loss = 0.0
regions = ['A', 'B']
x_max = 1000000.0
x_min = 0.0
S_min = 0.0
S_max = 2000000.0
gen_multiplier = [2, 2, 1, 1]
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

# Data Preprocessing
# The problem involves units of "1 million cubic meters" for storage/release limits
# and integer values like 80, 20 in the table. 
# Based on the magnitude of initial_storage (1,000,000), we infer the table units are 10^4 m^3.
# Scaling factor to convert table data to m^3:
SCALE = 10000.0

inflows = [val * SCALE for val in Table_1_Period_Data['Inflow']]
max_supply_A = [val * SCALE for val in Table_1_Period_Data['MaxSupplyA']]
max_supply_B = [val * SCALE for val in Table_1_Period_Data['MaxSupplyB']]
min_supply_A = [val * SCALE for val in Table_1_Period_Data['MinSupplyA']]
min_supply_B = [val * SCALE for val in Table_1_Period_Data['MinSupplyB']]

# Decision Variables
# R[t]: Total water release in period t
# RA[t]: Water supply to region A
# RB[t]: Water supply to region B
# S[t]: Storage at end of period t
# P[t]: Power generation in period t

R = model.addVars(T, lb=x_min, ub=x_max, vtype=GRB.CONTINUOUS, name="R")
RA = model.addVars(T, vtype=GRB.CONTINUOUS, name="RA")
RB = model.addVars(T, vtype=GRB.CONTINUOUS, name="RB")
S = model.addVars(T, lb=S_min, ub=S_max, vtype=GRB.CONTINUOUS, name="S")
P = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="P")

# Objective Function
# Maximize total power generation
model.setObjective(gp.quicksum(P[t] for t in range(T)), GRB.MAXIMIZE)

# Constraints

# Previous storage tracker, initialized with S0
current_prev_S = initial_storage

for t in range(T):
    # 1. Supply Demand Limits
    model.addConstr(RA[t] <= max_supply_A[t], name=f"MaxA_{t}")
    model.addConstr(RA[t] >= min_supply_A[t], name=f"MinA_{t}")
    model.addConstr(RB[t] <= max_supply_B[t], name=f"MaxB_{t}")
    model.addConstr(RB[t] >= min_supply_B[t], name=f"MinB_{t}")
    
    # 2. Supply Allocation Equality
    model.addConstr(R[t] == RA[t] + RB[t], name=f"SupplyEq_{t}")
    
    # 3. Water Balance Equation
    # S_t = S_{t-1} + I_t - R_t
    # Note: natural_loss is 0, so it is omitted
    model.addConstr(S[t] == current_prev_S + inflows[t] - R[t], name=f"Balance_{t}")
    
    # Update previous storage for next iteration (linking variables)
    current_prev_S = S[t]
    
    # 4. Power Generation Relation
    model.addConstr(P[t] == gen_multiplier[t] * R[t], name=f"Power_{t}")

# 5. Final Storage Constraint
# The storage at the end of period 4 (index 3) must be >= 200,000
model.addConstr(S[T-1] >= final_storage_min, name="FinalStorage")

# Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print("Optimal Solution Found:")
    for t in range(T):
        print(f"Period {t+1}: Release={R[t].X}, Storage={S[t].X}, Power={P[t].X}")
    
    final_objective = model.objVal
    print(f"Total Power Generation: {final_objective}")
    print(f"FinalAnswer=【{final_objective}】")
else:
    print("No solution found")