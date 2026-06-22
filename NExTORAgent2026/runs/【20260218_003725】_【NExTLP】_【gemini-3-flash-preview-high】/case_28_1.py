import gurobipy as gp
from gurobipy import GRB

# Define the model
model = gp.Model("SupplyChainOptimization")

# Plants: A1, A2
# Warehouses: B1, B2, B3, B4, B5, B6
# Users: C1, C2, C3, C4, C5, C6

# Parameters from Table C-16 and C-17
# Plant capacities
supply_cap = {1: 150000, 2: 200000}

# User demands
user_demand = {1: 50000, 2: 10000, 3: 40000, 4: 35000, 5: 60000, 6: 20000}

# Transportation costs from plants to warehouses (A_i to B_j)
cost_A_B = {
    (1, 1): 50, (1, 2): 50, (1, 3): 100, (1, 4): 20, (1, 5): 60, (1, 6): 40,
    (2, 2): 30, (2, 3): 50, (2, 4): 20, (2, 5): 40, (2, 6): 30
}

# Transportation costs from plants to users (A_i to C_k)
cost_A_C = {
    (1, 1): 100, (1, 3): 150, (1, 4): 200, (1, 6): 100,
    (2, 1): 200
}

# Transportation costs from warehouses to users (B_j to C_k)
cost_B_C = {
    (1, 2): 150, (1, 3): 50, (1, 4): 150, (1, 6): 100,
    # Alignment for B2 based on table structure analysis
    (2, 2): 100, (2, 3): 50, (2, 4): 50, (2, 5): 100, (2, 6): 50,
    (3, 2): 150, (3, 3): 200, (3, 5): 50, (3, 6): 150,
    # Assuming B4 supplies C6 based on preference
    (4, 6): 50,
    # From Table C-17
    (5, 1): 120, (5, 2): 60, (5, 3): 40, (5, 5): 30, (5, 6): 80,
    (6, 2): 40, (6, 4): 50, (6, 5): 60, (6, 6): 90
}

# Create decision variables
# Flow from plants to warehouses
w = model.addVars(cost_A_B.keys(), name="w")
# Flow from plants to users (direct supply)
z = model.addVars(cost_A_C.keys(), name="z")
# Flow from warehouses to users
y = model.addVars(cost_B_C.keys(), name="y")

# Binary variables for keeping open or building new warehouses
# u1, u4: original warehouses (1 if remains open, 0 if closed)
# u5, u6: candidate warehouses (1 if built, 0 if not)
u = model.addVars([1, 4, 5, 6], vtype=GRB.BINARY, name="u")
# B2 and B3 are existing and always open
u[2] = 1
u[3] = 1

# Binary variable for expanding B2
e2 = model.addVar(vtype=GRB.BINARY, name="e2")

# Objective function
# Transportation costs
transport_cost = gp.quicksum(cost_A_B[i, j] * w[i, j] for i, j in cost_A_B.keys()) + \
                 gp.quicksum(cost_A_C[i, k] * z[i, k] for i, k in cost_A_C.keys()) + \
                 gp.quicksum(cost_B_C[j, k] * y[j, k] for j, k in cost_B_C.keys())

# Investment and closure savings
investment_cost = 1200000 * u[5] + 400000 * u[6] + 300000 * e2 - 100000 * (1 - u[1]) - 50000 * (1 - u[4])

model.setObjective(transport_cost + investment_cost, GRB.MINIMIZE)

# Constraints
# 1. Plant Supply Capacity
for i in [1, 2]:
    model.addConstr(gp.quicksum(w[i, j] for j in range(1, 7) if (i, j) in cost_A_B) +
                    gp.quicksum(z[i, k] for k in range(1, 7) if (i, k) in cost_A_C) <= supply_cap[i])

# 2. Warehouse Flow Conservation (In = Out)
for j in range(1, 7):
    model.addConstr(gp.quicksum(w[i, j] for i in [1, 2] if (i, j) in cost_A_B) ==
                    gp.quicksum(y[j, k] for k in range(1, 7) if (j, k) in cost_B_C))

# 3. Warehouse Throughput Capacity
# B1
model.addConstr(gp.quicksum(w[i, 1] for i in [1, 2] if (i, 1) in cost_A_B) <= 70000 * u[1])
# B2
model.addConstr(gp.quicksum(w[i, 2] for i in [1, 2] if (i, 2) in cost_A_B) <= 50000 + 20000 * e2)
# B3
model.addConstr(gp.quicksum(w[i, 3] for i in [1, 2] if (i, 3) in cost_A_B) <= 100000)
# B4
model.addConstr(gp.quicksum(w[i, 4] for i in [1, 2] if (i, 4) in cost_A_B) <= 40000 * u[4])
# B5
model.addConstr(gp.quicksum(w[i, 5] for i in [1, 2] if (i, 5) in cost_A_B) <= 30000 * u[5])
# B6
model.addConstr(gp.quicksum(w[i, 6] for i in [1, 2] if (i, 6) in cost_A_B) <= 25000 * u[6])

# 4. User Monthly Demand
for k in range(1, 7):
    model.addConstr(gp.quicksum(z[i, k] for i in [1, 2] if (i, k) in cost_A_C) +
                    gp.quicksum(y[j, k] for j in range(1, 7) if (j, k) in cost_B_C) >= user_demand[k])

# 5. Maximum 4 Warehouses Open
model.addConstr(gp.quicksum(u[j] for j in [1, 2, 3, 4, 5, 6]) <= 4)

# Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print(f"Minimum total cost: {model.ObjVal}")
    print(f"FinalAnswer=【{int(model.ObjVal)}】")
else:
    print("Optimization was not successful.")