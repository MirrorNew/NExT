import gurobipy as gp
from gurobipy import GRB

# Define all parameter matrices and data inputs
num_generators = 3
num_substations = 4
num_periods = 5

# Generator parameters
P_min = [20, 30, 0]
P_max = [50, 80, 70]
Ramp_limit = [40, 30, 70]
unit_cost = [50, 60, 100]

# Substation demands D[substation][period]
# Table data:
# Substation 1: [40, 30, 60, 35, 50]
# Substation 2: [30, 30, 40, 25, 40]
# Substation 3: [50, 40, 50, 40, 30]
# Substation 4: [30, 20, 30, 30, 40]
demands = [
    [40, 30, 60, 35, 50],
    [30, 30, 40, 25, 40],
    [50, 40, 50, 40, 30],
    [30, 20, 30, 30, 40]
]

fixed_fee = 500

# Create the model
model = gp.Model("SouthChinaElectricPowerDispatching")

# 3. Create decision variables
# u[i, t]: binary status of unit i in period t (1=ON, 0=OFF)
u = model.addVars(num_generators, num_periods, vtype=GRB.BINARY, name="u")
# x[i, j, t]: assignment indicator (unit i supplies substation j in period t)
x = model.addVars(num_generators, num_substations, num_periods, vtype=GRB.BINARY, name="x")
# P[i, t]: total power output of unit i in period t (MW)
P = model.addVars(num_generators, num_periods, vtype=GRB.CONTINUOUS, lb=0, name="P")
# P_sub[i, j, t]: power delivered from unit i to substation j in period t (MW)
P_sub = model.addVars(num_generators, num_substations, num_periods, vtype=GRB.CONTINUOUS, lb=0, name="P_sub")

# 5. Set up the objective function
# Overall goal: Minimize total power generation cost + fixed fee
total_generation_cost = gp.quicksum(unit_cost[i] * P[i, t] for i in range(num_generators) for t in range(num_periods))
model.setObjective(total_generation_cost + fixed_fee, GRB.MINIMIZE)

# 6. Add all constraints
for t in range(num_periods):
    # C1, C2: Output limits using indicator constraints
    for i in range(num_generators):
        # Case u=1: Output must be between P_min and P_max
        model.addGenConstrIndicator(u[i, t], 1, P[i, t] >= P_min[i])
        model.addGenConstrIndicator(u[i, t], 1, P[i, t] <= P_max[i])
        # Case u=0: Output must be 0
        model.addGenConstrIndicator(u[i, t], 0, P[i, t] <= 0)

    # C3: Coupling u-P (Power balance Unit Output)
    for i in range(num_generators):
        model.addConstr(P[i, t] == gp.quicksum(P_sub[i, j, t] for j in range(num_substations)))

    # C4: Demand satisfaction for each substation
    for j in range(num_substations):
        model.addConstr(gp.quicksum(P_sub[i, j, t] for i in range(num_generators)) == demands[j][t])

    # C5, C6: Assignment linking and max substations per unit
    for i in range(num_generators):
        # Each unit can supply at most two substations per time period
        model.addConstr(gp.quicksum(x[i, j, t] for j in range(num_substations)) <= 2)
        for j in range(num_substations):
            # Cannot assign if unit is not ON
            model.addConstr(x[i, j, t] <= u[i, t])
            # Assignment linking using indicators
            # If x=1, power must be positive (use small epsilon); if x=0, power must be 0
            model.addGenConstrIndicator(x[i, j, t], 1, P_sub[i, j, t] >= 0.0001)
            model.addGenConstrIndicator(x[i, j, t], 0, P_sub[i, j, t] <= 0)

    # C11: Spare capacity requirement (At least two units started in any period)
    model.addConstr(gp.quicksum(u[i, t] for i in range(num_generators)) >= 2)

# C7, C8: Ramp-up and Ramp-down limits
# Ramp limits are on the "next period". In our 0-indexed code, t=1 to 4 are restricted by t-1.
# Feasibility note: Initial period output is not restricted by R if start-up is allowed to max.
for i in range(num_generators):
    for t in range(1, num_periods):
        model.addConstr(P[i, t] - P[i, t-1] <= Ramp_limit[i])
        model.addConstr(P[i, t-1] - P[i, t] <= Ramp_limit[i])

# C9: Fault tolerance (No four consecutive assignments)
# If supplied for 3 consecutive periods, cannot supply in the 4th period
for i in range(num_generators):
    for j in range(num_substations):
        for t in range(num_periods - 3):
            model.addConstr(x[i, j, t] + x[i, j, t+1] + x[i, j, t+2] + x[i, j, t+3] <= 3)

# C10: Unit 3 Maintenance requirement (Shutdown at least once in five hours)
model.addConstr(gp.quicksum(u[2, t] for t in range(num_periods)) <= 4)

# 7. Solve the model and print results
model.optimize()

if model.status == GRB.OPTIMAL:
    final_cost = model.objVal
    print(f"FinalAnswer=【{final_cost}】")
else:
    print("No optimal solution found.")