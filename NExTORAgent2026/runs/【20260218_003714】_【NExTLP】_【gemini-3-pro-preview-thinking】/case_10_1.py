import gurobipy as gp
from gurobipy import GRB

# Define the model
model = gp.Model("SouthChinaPowerDispatch")

# --- Parameters ---
# Generators
# Indices: 0: Unit 1, 1: Unit 2, 2: Unit 3
num_units = 3
P_min = [20, 30, 0]
P_max = [50, 80, 70]
R = [40, 30, 70]
c = [50, 60, 100]

# Substations and Demand
# D is provided as D[substation_index][time_index]
# Indices: 0..3 for substations 1..4
# Time indices: 0..4 for periods 1..5
D = [
    [40, 30, 60, 35, 50],
    [30, 30, 40, 25, 40],
    [50, 40, 50, 40, 30],
    [30, 20, 30, 30, 40]
]
num_substations = 4
num_periods = 5

fixed_fee = 500

# --- Decision Variables ---
# u[i, t]: Unit i ON/OFF in period t
u = model.addVars(num_units, num_periods, vtype=GRB.BINARY, name="u")

# P[i, t]: Total output of unit i in period t
P = model.addVars(num_units, num_periods, vtype=GRB.CONTINUOUS, lb=0, name="P")

# x[i, j, t]: Unit i supplies substation j in period t
x = model.addVars(num_units, num_substations, num_periods, vtype=GRB.BINARY, name="x")

# P_flow[i, j, t]: Power from unit i to substation j in period t
P_flow = model.addVars(num_units, num_substations, num_periods, vtype=GRB.CONTINUOUS, lb=0, name="P_flow")

# --- Objective Function ---
# Minimize total generation cost + fixed fee
obj_expr = gp.quicksum(c[i] * P[i, t] for i in range(num_units) for t in range(num_periods)) + fixed_fee
model.setObjective(obj_expr, GRB.MINIMIZE)

# --- Constraints ---

# 1. & 2. Unit Output Limits (Min and Max)
# C1: P_{i,t} >= P_min * u_{i,t}
# C2: P_{i,t} <= P_max * u_{i,t}
for i in range(num_units):
    for t in range(num_periods):
        model.addConstr(P[i, t] >= P_min[i] * u[i, t], name=f"MinOutput_{i}_{t}")
        model.addConstr(P[i, t] <= P_max[i] * u[i, t], name=f"MaxOutput_{i}_{t}")

# 3. Output Aggregation
# C3: P_{i,t} = sum(P_{i,j,t} over j)
for i in range(num_units):
    for t in range(num_periods):
        model.addConstr(P[i, t] == gp.quicksum(P_flow[i, j, t] for j in range(num_substations)), name=f"Agg_{i}_{t}")

# 4. Demand Satisfaction
# C4: sum(P_{i,j,t} over i) = D_{j,t}
for j in range(num_substations):
    for t in range(num_periods):
        model.addConstr(gp.quicksum(P_flow[i, j, t] for i in range(num_units)) == D[j][t], name=f"Demand_{j}_{t}")

# 5. Assignment Linking
# C5: P_{i,j,t} <= D_{j,t} * x_{i,j,t}
for i in range(num_units):
    for j in range(num_substations):
        for t in range(num_periods):
            model.addConstr(P_flow[i, j, t] <= D[j][t] * x[i, j, t], name=f"Link_{i}_{j}_{t}")

# 6. Max Two Substations per Unit
# C6: sum(x_{i,j,t} over j) <= 2
for i in range(num_units):
    for t in range(num_periods):
        model.addConstr(gp.quicksum(x[i, j, t] for j in range(num_substations)) <= 2, name=f"MaxSub_{i}_{t}")

# 7. & 8. Ramp-Up / Ramp-Down Limits
# C7, C8: |P_{i,t} - P_{i,t-1}| <= R_i
# Initial state: P_{i,0_prev} = 0. So for t=0 (Period 1): P_{i,0} <= R_i (since P >= 0)
for i in range(num_units):
    # Period 1 (index 0)
    model.addConstr(P[i, 0] <= R[i], name=f"RampInit_{i}")
    # Periods 2-5 (indices 1-4)
    for t in range(1, num_periods):
        model.addConstr(P[i, t] - P[i, t-1] <= R[i], name=f"RampUp_{i}_{t}")
        model.addConstr(P[i, t-1] - P[i, t] <= R[i], name=f"RampDown_{i}_{t}")

# 9. No Four Consecutive Assignments
# C9: sum(x_{i,j,tau} for tau in [t-3, t]) <= 3 for t in [3, 4] (periods 4, 5)
for i in range(num_units):
    for j in range(num_substations):
        # Check window ending at index 3 (Period 4) -> indices 0,1,2,3
        model.addConstr(gp.quicksum(x[i, j, tau] for tau in range(4)) <= 3, name=f"Consec_4_{i}_{j}")
        # Check window ending at index 4 (Period 5) -> indices 1,2,3,4
        model.addConstr(gp.quicksum(x[i, j, tau] for tau in range(1, 5)) <= 3, name=f"Consec_5_{i}_{j}")

# 10. Unit 3 Maintenance
# Unit 3 (index 2) off at least once -> sum(u[2,t]) <= 4
model.addConstr(gp.quicksum(u[2, t] for t in range(num_periods)) <= 4, name="Maintenance_Unit3")

# 11. Spare Capacity (Min Units Started)
# C11: sum(u_{i,t} over i) >= 2
for t in range(num_periods):
    model.addConstr(gp.quicksum(u[i, t] for i in range(num_units)) >= 2, name=f"MinUnits_{t}")

# --- Solve ---
model.optimize()

# --- Output ---
if model.status == GRB.OPTIMIZED:
    print(f"Optimal Objective Value: {model.objVal}")
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found.")