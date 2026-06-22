import gurobipy as gp
from gurobipy import GRB

# ===============================
# 1. Parameters (from Parameters List)
# ===============================

T = 15                 # planning horizon
S = 4                  # number of stores
alpha = 0.3            # safety stock fraction
min_orders_CW_RW1 = 10 # minimum orders from CW to RW1

# Table_1_Inventory_Demand
Table_1_Inventory_Demand = [
    {'Entity': 'Central Warehouse', 'Initial Stock (Units)': 1200, 'Demand per Period (Units)': None},
    {'Entity': 'Regional Warehouse 1', 'Initial Stock (Units)': 500, 'Demand per Period (Units)': None},
    {'Entity': 'Regional Warehouse 2', 'Initial Stock (Units)': 400, 'Demand per Period (Units)': None},
    {'Entity': 'Retail Store 1', 'Initial Stock (Units)': 350, 'Demand per Period (Units)': 50},
    {'Entity': 'Retail Store 2', 'Initial Stock (Units)': 450, 'Demand per Period (Units)': 60},
    {'Entity': 'Retail Store 3', 'Initial Stock (Units)': 500, 'Demand per Period (Units)': 70},
    {'Entity': 'Retail Store 4', 'Initial Stock (Units)': 600, 'Demand per Period (Units)': 80},
]

# Table_2_Transport_Route_Costs
Table_2_Transport_Route_Costs = [
    {'Transport Route': 'Central Warehouse → Regional Warehouse 1',
     'Transport Costs (Euro/Unit)': 0.55,
     'Maximum Transport Capacity per Period (Units)': 1000},
    {'Transport Route': 'Central Warehouse → Regional Warehouse 2',
     'Transport Costs (Euro/Unit)': 0.22,
     'Maximum Transport Capacity per Period (Units)': 1000},
    {'Transport Route': 'Regional warehouse 1 → Retail store 1',
     'Transport Costs (Euro/Unit)': 0.22,
     'Maximum Transport Capacity per Period (Units)': 500},
    {'Transport Route': 'Regional warehouse 1 → Retail store 2',
     'Transport Costs (Euro/Unit)': 0.2,
     'Maximum Transport Capacity per Period (Units)': 500},
    {'Transport Route': 'Regional warehouse 1 → Retail store 3',
     'Transport Costs (Euro/Unit)': 0.32,
     'Maximum Transport Capacity per Period (Units)': 500},
    {'Transport Route': 'Regional warehouse 1 → Retail store 4',
     'Transport Costs (Euro/Unit)': 0.38,
     'Maximum Transport Capacity per Period (Units)': 500},
    {'Transport Route': 'Regional warehouse 2 → Retail store 1',
     'Transport Costs (Euro/Unit)': 0.68,
     'Maximum Transport Capacity per Period (Units)': 500},
    {'Transport Route': 'Regional warehouse 2 → Retail store 2',
     'Transport Costs (Euro/Unit)': 0.52,
     'Maximum Transport Capacity per Period (Units)': 500},
    {'Transport Route': 'Regional warehouse 2 → Retail store 3',
     'Transport Costs (Euro/Unit)': 0.34,
     'Maximum Transport Capacity per Period (Units)': 500},
    {'Transport Route': 'Regional warehouse 2 → Retail store 4',
     'Transport Costs (Euro/Unit)': 0.1,
     'Maximum Transport Capacity per Period (Units)': 500},
]

# Table_3_Costs
Table_3_Costs = [
    {'Cost type': 'Ordering cost', 'Value': 30},
    {'Cost type': 'Warehouse holding cost', 'Value': 0.2},
    {'Cost type': 'Retail store holding costs', 'Value': 0.6},
]

# ===============================
# 2. Derived parameters / data structures
# ===============================

# Index sets
periods = range(1, T + 1)
warehouses = [1, 2]              # regional warehouses
stores = [1, 2, 3, 4]            # retail stores

# Initial inventories and demands from Table_1_Inventory_Demand
I0_C = Table_1_Inventory_Demand[0]['Initial Stock (Units)']
I0_R = {1: Table_1_Inventory_Demand[1]['Initial Stock (Units)'],
        2: Table_1_Inventory_Demand[2]['Initial Stock (Units)']}
I0_S = {1: Table_1_Inventory_Demand[3]['Initial Stock (Units)'],
        2: Table_1_Inventory_Demand[4]['Initial Stock (Units)'],
        3: Table_1_Inventory_Demand[5]['Initial Stock (Units)'],
        4: Table_1_Inventory_Demand[6]['Initial Stock (Units)']}

demand = {
    1: Table_1_Inventory_Demand[3]['Demand per Period (Units)'],
    2: Table_1_Inventory_Demand[4]['Demand per Period (Units)'],
    3: Table_1_Inventory_Demand[5]['Demand per Period (Units)'],
    4: Table_1_Inventory_Demand[6]['Demand per Period (Units)'],
}

# Costs from Table_3_Costs
ordering_cost = next(item['Value'] for item in Table_3_Costs
                     if item['Cost type'] == 'Ordering cost')
warehouse_holding_cost = next(item['Value'] for item in Table_3_Costs
                              if item['Cost type'] == 'Warehouse holding cost')
store_holding_cost = next(item['Value'] for item in Table_3_Costs
                          if item['Cost type'] == 'Retail store holding costs')

# Transport costs and capacities
# Central → Regional
c_CR = {1: Table_2_Transport_Route_Costs[0]['Transport Costs (Euro/Unit)'],
        2: Table_2_Transport_Route_Costs[1]['Transport Costs (Euro/Unit)']}
cap_CR = {1: Table_2_Transport_Route_Costs[0]['Maximum Transport Capacity per Period (Units)'],
          2: Table_2_Transport_Route_Costs[1]['Maximum Transport Capacity per Period (Units)']}

# Regional → Store
c_RS = {}
cap_RS = {}
# Rows 2..9 in Table_2_Transport_Route_Costs are regional→store
# Order: RW1→S1, RW1→S2, RW1→S3, RW1→S4, RW2→S1, RW2→S2, RW2→S3, RW2→S4
for idx, (i, j) in enumerate([(1, 1), (1, 2), (1, 3), (1, 4),
                              (2, 1), (2, 2), (2, 3), (2, 4)]):
    row = Table_2_Transport_Route_Costs[2 + idx]
    c_RS[i, j] = row['Transport Costs (Euro/Unit)']
    cap_RS[i, j] = row['Maximum Transport Capacity per Period (Units)']

# Big-M for indicator linking z and y (must be at least as big as capacity)
# Using the max of Cap^{CR}_i as a valid bound
M = max(cap_CR.values())

# ===============================
# 3. Create model
# ===============================

model = gp.Model("Morning_Light_Supply_Chain")

# ===============================
# 4. Decision variables
# ===============================

# y[i,t]: binary order placement decision at regional warehouse i in period t
y = model.addVars(warehouses, periods, vtype=GRB.BINARY, name="y")

# z[i,t]: shipment from central warehouse to regional warehouse i in period t
z = model.addVars(warehouses, periods, lb=0.0, vtype=GRB.CONTINUOUS, name="z")

# w[i,j,t]: shipment from regional warehouse i to store j in period t
w = model.addVars(warehouses, stores, periods, lb=0.0,
                  vtype=GRB.CONTINUOUS, name="w")

# Inventories
# I_C[t]: central warehouse inventory at end of period t
I_C = model.addVars(periods, lb=0.0, vtype=GRB.CONTINUOUS, name="I_C")

# I_R[i,t]: regional warehouse inventory at end of period t
I_R = model.addVars(warehouses, periods, lb=0.0,
                    vtype=GRB.CONTINUOUS, name="I_R")

# I_S[j,t]: store inventory at end of period t
I_S = model.addVars(stores, periods, lb=0.0,
                    vtype=GRB.CONTINUOUS, name="I_S")

# ===============================
# 5. Constraints
# ===============================

# 5.1 Central warehouse inventory balance
for t in periods:
    if t == 1:
        prev_I_C = I0_C
    else:
        prev_I_C = I_C[t - 1]
    model.addConstr(
        I_C[t] == prev_I_C - gp.quicksum(z[i, t] for i in warehouses),
        name=f"CentralBalance_{t}"
    )

# 5.2 Regional warehouse inventory balance
for i in warehouses:
    for t in periods:
        if t == 1:
            prev_I_R = I0_R[i]
        else:
            prev_I_R = I_R[i, t - 1]
        model.addConstr(
            I_R[i, t] == prev_I_R + z[i, t] - gp.quicksum(w[i, j, t] for j in stores),
            name=f"RegionalBalance_{i}_{t}"
        )

# 5.3 Store inventory balance
for j in stores:
    for t in periods:
        if t == 1:
            prev_I_S = I0_S[j]
        else:
            prev_I_S = I_S[j, t - 1]
        model.addConstr(
            I_S[j, t] == prev_I_S
            + gp.quicksum(w[i, j, t] for i in warehouses)
            - demand[j],
            name=f"StoreBalance_{j}_{t}"
        )

# 5.4 Safety stock constraints at stores: I_S[j,t] >= alpha * d_j
for j in stores:
    for t in periods:
        model.addConstr(
            I_S[j, t] >= alpha * demand[j],
            name=f"SafetyStock_{j}_{t}"
        )

# 5.5 Route capacity constraints

# Central → Regional: 0 <= z[i,t] <= Cap^{CR}_i
for i in warehouses:
    for t in periods:
        model.addConstr(
            z[i, t] <= cap_CR[i],
            name=f"Cap_CR_{i}_{t}"
        )

# Regional → Store: 0 <= w[i,j,t] <= Cap^{RS}_{i,j}
for i in warehouses:
    for j in stores:
        for t in periods:
            model.addConstr(
                w[i, j, t] <= cap_RS[i, j],
                name=f"Cap_RS_{i}_{j}_{t}"
            )

# 5.6 Order–quantity linking: use indicator constraints (no big-M in the model body)
# If y[i,t] == 0 → z[i,t] == 0
for i in warehouses:
    for t in periods:
        model.addGenConstrIndicator(
            y[i, t], 0, z[i, t] == 0,
            name=f"IndLink_y0_{i}_{t}"
        )
        # If y[i,t] == 1 → z[i,t] <= M
        # (Upper bound already enforced by capacity and M >= capacity; here we link logically)
        model.addGenConstrIndicator(
            y[i, t], 1, z[i, t] <= M,
            name=f"IndLink_y1_{i}_{t}"
        )

# 5.7 Minimum number of orders to warehouse 1: sum_t y[1,t] >= min_orders_CW_RW1
model.addConstr(
    gp.quicksum(y[1, t] for t in periods) >= min_orders_CW_RW1,
    name="MinOrders_RW1"
)

# ===============================
# 6. Objective function
# ===============================

# Components:
#  - Ordering cost: sum_i,t 30 * y[i,t]
ordering_cost_expr = ordering_cost * gp.quicksum(y[i, t] for i in warehouses for t in periods)

#  - Warehouse holding cost: 0.2 * sum_t (I_C[t] + sum_i I_R[i,t])
warehouse_holding_expr = warehouse_holding_cost * (
    gp.quicksum(I_C[t] for t in periods) +
    gp.quicksum(I_R[i, t] for i in warehouses for t in periods)
)

#  - Store holding cost: 0.6 * sum_t,j I_S[j,t]
store_holding_expr = store_holding_cost * gp.quicksum(I_S[j, t] for j in stores for t in periods)

#  - Transportation cost central→regional: sum_i,t c_CR[i] * z[i,t]
transport_CR_expr = gp.quicksum(c_CR[i] * z[i, t] for i in warehouses for t in periods)

#  - Transportation cost regional→store: sum_i,j,t c_RS[i,j] * w[i,j,t]
transport_RS_expr = gp.quicksum(c_RS[i, j] * w[i, j, t]
                                for i in warehouses for j in stores for t in periods)

total_cost = ordering_cost_expr + warehouse_holding_expr + store_holding_expr \
             + transport_CR_expr + transport_RS_expr

model.setObjective(total_cost, GRB.MINIMIZE)

# ===============================
# 7. Solve model
# ===============================

model.optimize()

# ===============================
# 8. Print results
# ===============================

if model.Status == GRB.OPTIMAL:
    print(f"Optimal objective (minimum total cost) = {model.ObjVal:.4f}")

    # Example: show order decisions and shipments (optional detailed output)
    for i in warehouses:
        for t in periods:
            if y[i, t].X > 0.5:
                print(f"Order placed: warehouse {i}, period {t}, "
                      f"quantity from central = {z[i, t].X:.2f}")

    # Final inventories at each location at the end of horizon
    print("\nFinal inventories at end of period T:")
    print(f"Central warehouse: {I_C[T].X:.2f}")
    for i in warehouses:
        print(f"Regional warehouse {i}: {I_R[i, T].X:.2f}")
    for j in stores:
        print(f"Store {j}: {I_S[j, T].X:.2f}")

    # FinalAnswer must be the minimum total cost
    print(f"FinalAnswer=【{model.ObjVal:.4f}】")
else:
    print("No optimal solution found.")
    # In case no optimal solution, still print FinalAnswer as NaN
    print("FinalAnswer=【NaN】")