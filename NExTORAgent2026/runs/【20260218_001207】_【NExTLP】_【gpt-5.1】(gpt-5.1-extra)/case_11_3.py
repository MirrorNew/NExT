import gurobipy as gp

# Complete Gurobi model for the given MILP problem
# FinalAnswer is the optimal total supply chain cost (objective value)

# =========================
# 1. Define parameters
# =========================
num_factories = 2
factories = ['Shenzhen Factory 1', 'Vietnam Factory']

num_warehouses = 3
warehouses = ['City A', 'City B', 'City C']

num_customers = 3
customers_by_name = ['Singapore', 'Malaysia', 'Philippines']

fixed_cost_warehouse = [500, 400, 300]
required_cost_reduction_percentage = 0.15

factory_capacity = [1000, 800]            # [Shenzhen, Vietnam]
customer_demand = [500, 700, 500]         # [Customer 1, 2, 3]

# From Parameters List (warehouse→customer)
transport_cost_warehouse_to_customer = [
    [3, 4, None],   # City A → [C1, C2, C3]
    [None, 3, 3],   # City B → [C1, C2, C3]
    [3, 5, 2]       # City C → [C1, C2, C3]
]

# From Parameters List (factory→warehouse) as given
transport_cost_factory_to_warehouse_raw = [
    [2, None],
    [4, 1],
    [3, 2]
]
# We will not change any numeric value; we just store the raw list
# and will index it only where it makes sense in the code.

factory_to_warehouse_allowed = [
    [1, 1, 1],  # Shenzhen → A,B,C
    [0, 1, 1]   # Vietnam  → -,B,C
]

warehouse_to_customer_allowed = [
    [1, 1, 0],  # City A → C1,C2
    [0, 1, 1],  # City B → C2,C3
    [1, 1, 1]   # City C → C1,C2,C3
]

min_warehouses_open = 2

# Big-M: use a safe upper bound (≥ total capacity and total demand)
total_demand = sum(customer_demand)
total_capacity = sum(factory_capacity)
M = max(total_demand, total_capacity)

# =========================
# 2. Create model
# =========================
model = gp.Model("SupplyChain_Warehouse_Location_and_Routing")

# =========================
# 3. Decision variables
# =========================

# Warehouse open variables: y[i] ∈ {0,1}
y = model.addVars(
    range(num_warehouses),
    vtype=gp.GRB.BINARY,
    name="y"
)

# Factory → Warehouse shipment variables: x[k,i] ≥ 0 (only if allowed)
x = {}
for k in range(num_factories):
    for i in range(num_warehouses):
        if factory_to_warehouse_allowed[k][i] == 1:
            x[k, i] = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS,
                                   name=f"x_{k}_{i}")
        # Forbidden arcs: no variable

# Warehouse → Customer shipment variables: z[i,j] ≥ 0 (only if allowed)
z = {}
for i in range(num_warehouses):
    for j in range(num_customers):
        if warehouse_to_customer_allowed[i][j] == 1:
            z[i, j] = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS,
                                   name=f"z_{i}_{j}")
        # Forbidden arcs: no variable

model.update()

# =========================
# 4. Objective function
# Minimize total cost
# =========================
obj = gp.LinExpr()

# Fixed warehouse costs
for i in range(num_warehouses):
    obj.addTerms(fixed_cost_warehouse[i], y[i])

# Factory → Warehouse transportation costs
# We must strictly use transport_cost_factory_to_warehouse_raw values.
# We interpret:
#   - Shenzhen (k=0) to A,B,C from column 0 of the 3 rows
#   - Vietnam  (k=1) to B,C from column 1 of rows 1 and 2
for k in range(num_factories):
    for i in range(num_warehouses):
        if (k, i) not in x:
            continue
        cost_fw = None
        if k == 0:
            # Shenzhen: use column 0 of raw rows (indices 0,1,2) for A,B,C
            cost_fw = transport_cost_factory_to_warehouse_raw[i][0]
        else:
            # Vietnam: only B,C allowed, use column 1 of raw rows 1,2
            if i == 1:
                cost_fw = transport_cost_factory_to_warehouse_raw[1][1]
            elif i == 2:
                cost_fw = transport_cost_factory_to_warehouse_raw[2][1]
        if cost_fw is not None:
            obj.addTerms(cost_fw, x[k, i])

# Warehouse → Customer transportation costs
for i in range(num_warehouses):
    for j in range(num_customers):
        if (i, j) not in z:
            continue
        cost_wc = transport_cost_warehouse_to_customer[i][j]
        if cost_wc is not None:
            obj.addTerms(cost_wc, z[i, j])

model.setObjective(obj, gp.GRB.MINIMIZE)

# =========================
# 5. Constraints
# =========================

# Factory capacity constraints
# Shenzhen (k = 0)
model.addConstr(
    gp.quicksum(x[0, i] for i in range(num_warehouses) if (0, i) in x)
    <= factory_capacity[0],
    name="Shenzhen_capacity"
)

# Vietnam (k = 1)
model.addConstr(
    gp.quicksum(x[1, i] for i in range(num_warehouses) if (1, i) in x)
    <= factory_capacity[1],
    name="Vietnam_capacity"
)

# Customer demand satisfaction
# Customer 1 (Singapore, j = 0)
model.addConstr(
    gp.quicksum(z[i, 0] for i in range(num_warehouses) if (i, 0) in z)
    == customer_demand[0],
    name="Demand_C1"
)

# Customer 2 (Malaysia, j = 1)
model.addConstr(
    gp.quicksum(z[i, 1] for i in range(num_warehouses) if (i, 1) in z)
    == customer_demand[1],
    name="Demand_C2"
)

# Customer 3 (Philippines, j = 2)
model.addConstr(
    gp.quicksum(z[i, 2] for i in range(num_warehouses) if (i, 2) in z)
    == customer_demand[2],
    name="Demand_C3"
)

# Warehouse flow conservation:
# inbound (from factories) = outbound (to customers) for each warehouse
for i in range(num_warehouses):
    inbound = gp.quicksum(x[k, i] for k in range(num_factories) if (k, i) in x)
    outbound = gp.quicksum(z[i, j] for j in range(num_customers) if (i, j) in z)
    model.addConstr(inbound == outbound, name=f"Flow_warehouse_{i}")

# Activation linking (Big-M): if y[i] = 0 then no flow
for i in range(num_warehouses):
    # inbound
    for k in range(num_factories):
        if (k, i) in x:
            model.addConstr(x[k, i] <= M * y[i], name=f"Link_in_{k}_{i}")
    # outbound
    for j in range(num_customers):
        if (i, j) in z:
            model.addConstr(z[i, j] <= M * y[i], name=f"Link_out_{i}_{j}")

# Policy constraints
# Warehouse B (index 1) must be opened
model.addConstr(y[1] == 1, name="Warehouse_B_must_open")

# At least two warehouses must be opened
model.addConstr(
    gp.quicksum(y[i] for i in range(num_warehouses)) >= min_warehouses_open,
    name="Min_warehouses_open"
)

# =========================
# 6. Optimize
# =========================
model.optimize()

# =========================
# 7. Print results and FinalAnswer
# =========================
if model.status == gp.GRB.OPTIMAL:
    optimal_total_cost = model.objVal

    # Optional detailed output
    print(f"Optimal objective value (total cost): {optimal_total_cost:.2f}")
    print("Warehouse open decisions:")
    for i in range(num_warehouses):
        print(f"  {warehouses[i]}: open = {int(round(y[i].X))}")

    print("Factory → Warehouse shipments:")
    for (k, i), var in x.items():
        if var.X > 1e-6:
            print(f"  {factories[k]} → {warehouses[i]}: {var.X:.2f}")

    print("Warehouse → Customer shipments:")
    for (i, j), var in z.items():
        if var.X > 1e-6:
            print(f"  {warehouses[i]} → {customers_by_name[j]}: {var.X:.2f}")

    # Required final line:
    print(f"FinalAnswer=【{optimal_total_cost:.2f}】")
else:
    print(f"Optimization ended with status {model.status}")
    print("FinalAnswer=【NaN】")