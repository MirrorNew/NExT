import gurobipy as gp
from gurobipy import GRB

# Parameters
T = 15  # periods
I = [1, 2]  # regional warehouses
J = [1, 2, 3, 4]  # retail stores

# Cost parameters from Table_3_Costs
ordering_cost = 30  # Euro/time
warehouse_holding_cost = 0.2  # Euro/period Unit
store_holding_cost = 0.6  # Euro/period Unit

# Initial inventories from Table_1_Inventory_Demand
I_C0 = 1200  # Central Warehouse initial stock
I_R0 = {1: 500, 2: 400}  # Regional warehouse initial stocks
I_S0 = {1: 350, 2: 450, 3: 500, 4: 600}  # Retail store initial stocks

# Demand per period for retail stores from Table_1_Inventory_Demand
demand = {1: 50, 2: 60, 3: 70, 4: 80}  # units per period

# Safety stock parameter
alpha = 0.3  # 30% safety stock requirement
safety_stock = {j: alpha * demand[j] for j in J}  # safety stock for each store

# Transport costs and capacities from Table_2_Transport_Route_Costs
# Central to Regional
transport_cost_CR = {1: 0.55, 2: 0.22}  # Euro/unit
capacity_CR = {1: 1000, 2: 1000}  # units per period

# Regional to Store
transport_cost_RS = {
    (1, 1): 0.22, (1, 2): 0.20, (1, 3): 0.32, (1, 4): 0.38,
    (2, 1): 0.68, (2, 2): 0.52, (2, 3): 0.34, (2, 4): 0.10
}
capacity_RS = {
    (1, 1): 500, (1, 2): 500, (1, 3): 500, (1, 4): 500,
    (2, 1): 500, (2, 2): 500, (2, 3): 500, (2, 4): 500
}

# Minimum orders constraint
min_orders_CW_RW1 = 10

# Big M for order-quantity linking
M = 1000  # sufficiently large number

# Create model
model = gp.Model("MorningLight_SupplyChain")

# Decision Variables
y = {}  # y[i,t] - binary order decision
z = {}  # z[i,t] - replenishment quantity from central to regional
w = {}  # w[i,j,t] - shipment from regional to store
I_C = {}  # I_C[t] - central warehouse end-of-period inventory
I_R = {}  # I_R[i,t] - regional warehouse end-of-period inventory
I_S = {}  # I_S[j,t] - retail store end-of-period inventory

for i in I:
    for t in range(1, T+1):
        y[i, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{t}")
        z[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=capacity_CR[i], name=f"z_{i}_{t}")
        for j in J:
            w[i, j, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=capacity_RS[i, j], name=f"w_{i}_{j}_{t}")

for t in range(1, T+1):
    I_C[t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"I_C_{t}")
    for i in I:
        I_R[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"I_R_{i}_{t}")
    for j in J:
        I_S[j, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"I_S_{j}_{t}")

# Set objective function
obj = gp.quicksum(ordering_cost * y[i, t] for i in I for t in range(1, T+1))
obj += gp.quicksum(transport_cost_CR[i] * z[i, t] for i in I for t in range(1, T+1))
obj += gp.quicksum(transport_cost_RS[i, j] * w[i, j, t] for i in I for j in J for t in range(1, T+1))
obj += warehouse_holding_cost * gp.quicksum(I_C[t] for t in range(1, T+1))
obj += warehouse_holding_cost * gp.quicksum(I_R[i, t] for i in I for t in range(1, T+1))
obj += store_holding_cost * gp.quicksum(I_S[j, t] for j in J for t in range(1, T+1))

model.setObjective(obj, GRB.MINIMIZE)

# Constraints

# 1. Order-quantity linking: z[i,t] <= M * y[i,t]
for i in I:
    for t in range(1, T+1):
        model.addConstr(z[i, t] <= M * y[i, t], name=f"order_link_{i}_{t}")

# 2. Central warehouse inventory balance
# For t=1: I_C[1] = I_C0 - sum_i z[i,1]
model.addConstr(I_C[1] == I_C0 - gp.quicksum(z[i, 1] for i in I), name="central_balance_1")
# For t>1: I_C[t] = I_C[t-1] - sum_i z[i,t]
for t in range(2, T+1):
    model.addConstr(I_C[t] == I_C[t-1] - gp.quicksum(z[i, t] for i in I), name=f"central_balance_{t}")

# 3. Regional warehouse inventory balance
for i in I:
    # For t=1: I_R[i,1] = I_R0[i] + z[i,1] - sum_j w[i,j,1]
    model.addConstr(I_R[i, 1] == I_R0[i] + z[i, 1] - gp.quicksum(w[i, j, 1] for j in J), name=f"regional_balance_{i}_1")
    # For t>1: I_R[i,t] = I_R[i,t-1] + z[i,t] - sum_j w[i,j,t]
    for t in range(2, T+1):
        model.addConstr(I_R[i, t] == I_R[i, t-1] + z[i, t] - gp.quicksum(w[i, j, t] for j in J), name=f"regional_balance_{i}_{t}")

# 4. Retail store inventory balance
for j in J:
    # For t=1: I_S[j,1] = I_S0[j] + sum_i w[i,j,1] - demand[j]
    model.addConstr(I_S[j, 1] == I_S0[j] + gp.quicksum(w[i, j, 1] for i in I) - demand[j], name=f"store_balance_{j}_1")
    # For t>1: I_S[j,t] = I_S[j,t-1] + sum_i w[i,j,t] - demand[j]
    for t in range(2, T+1):
        model.addConstr(I_S[j, t] == I_S[j, t-1] + gp.quicksum(w[i, j, t] for i in I) - demand[j], name=f"store_balance_{j}_{t}")

# 5. No stockouts at stores (implicitly covered by safety stock constraint which is stricter)

# 6. Safety stock requirement: I_S[j,t] >= 0.3 * demand[j]
for j in J:
    for t in range(1, T+1):
        model.addConstr(I_S[j, t] >= safety_stock[j], name=f"safety_stock_{j}_{t}")

# 7. Transport capacity limits (already set as variable bounds, but adding explicit constraints for clarity)
for i in I:
    for t in range(1, T+1):
        model.addConstr(z[i, t] <= capacity_CR[i], name=f"capacity_CR_{i}_{t}")
        for j in J:
            model.addConstr(w[i, j, t] <= capacity_RS[i, j], name=f"capacity_RS_{i}_{j}_{t}")

# 8. Minimum orders to warehouse 1: sum_t y[1,t] >= 10
model.addConstr(gp.quicksum(y[1, t] for t in range(1, T+1)) >= min_orders_CW_RW1, name="min_orders_RW1")

# Solve the model
model.optimize()

# Check if solution is optimal
if model.status == GRB.OPTIMAL:
    total_cost = model.ObjVal
    print(f"Optimal solution found!")
    print(f"Total cost: {total_cost:.2f} EUR")
    
    # Print order decisions for warehouse 1
    orders_rw1 = sum(y[1, t].X for t in range(1, T+1))
    print(f"Number of orders from central to warehouse 1: {int(orders_rw1)}")
    
    # Final answer
    print(f"FinalAnswer=【{total_cost:.2f}】")
else:
    print(f"Model status: {model.status}")
    print("No optimal solution found.")
    print(f"FinalAnswer=【No optimal solution】")