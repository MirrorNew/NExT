import gurobipy as gp
from gurobipy import GRB

# Define the model
model = gp.Model("SupplyChainOptimization")

# 1. Parameter Definitions based on Parameters List
factories = ['Shenzhen', 'Vietnam']
warehouses = ['CityA', 'CityB', 'CityC']
customers = ['Singapore', 'Malaysia', 'Philippines']

# Mappings for data consistency
factory_map = {'Shenzhen': 'ShenzhenFactory', 'Vietnam': 'VietnamFactory'}
customer_map = {'Singapore': 'Customer1', 'Malaysia': 'Customer2', 'Philippines': 'Customer3'}
warehouse_code_map = {'CityA': 'A', 'CityB': 'B', 'CityC': 'C'}

# Fixed Operating Costs (mapped from A, B, C to CityA, CityB, CityC)
fixed_costs_data = {'A': 500, 'B': 400, 'C': 300}
fixed_operating_cost = {w: fixed_costs_data[warehouse_code_map[w]] for w in warehouses}

# Capacity and Demand Data
table_1 = {
    'ShenzhenFactory': 1000, 
    'VietnamFactory': 800, 
    'Customer1': 500, 
    'Customer2': 700, 
    'Customer3': 500
}
factory_capacity = {f: table_1[factory_map[f]] for f in factories}
customer_demand = {c: table_1[customer_map[c]] for c in customers}

# Transportation Costs: Warehouse -> Customer
# Data: {'CityA': {'Customer1': 3, ...}, ...}
table_2 = {
    'CityA': {'Customer1': 3, 'Customer2': 4, 'Customer3': None}, 
    'CityB': {'Customer1': None, 'Customer2': 3, 'Customer3': 3}, 
    'CityC': {'Customer1': 3, 'Customer2': 5, 'Customer3': 2}
}
cost_wc = {}
for w in warehouses:
    for c in customers:
        val = table_2[w].get(customer_map[c])
        cost_wc[(w, c)] = val # Can be None

# Transportation Costs: Factory -> Warehouse
# Data: {'CityA': {'Shenzhen': 2, 'Vietnam': None}, ...}
table_3 = {
    'CityA': {'Shenzhen': 2, 'Vietnam': None}, 
    'CityB': {'Shenzhen': 4, 'Vietnam': 1}, 
    'CityC': {'Shenzhen': 3, 'Vietnam': 2}
}
cost_fw = {}
for f in factories:
    for w in warehouses:
        # Note: Structure is table_3[w][f]
        val = table_3[w].get(f)
        cost_fw[(f, w)] = val # Can be None

# 2. Decision Variables

# Binary variables for opening warehouses
y = model.addVars(warehouses, vtype=GRB.BINARY, name="y")

# Continuous variables for flow from Factory to Warehouse
x_fw = model.addVars(factories, warehouses, vtype=GRB.CONTINUOUS, lb=0, name="x_fw")

# Continuous variables for flow from Warehouse to Customer
x_wc = model.addVars(warehouses, customers, vtype=GRB.CONTINUOUS, lb=0, name="x_wc")

# 3. Objective Function
# Minimize Total Cost = Fixed Cost + Transport(Factory->Warehouse) + Transport(Warehouse->Customer)
total_fixed_cost = gp.quicksum(fixed_operating_cost[w] * y[w] for w in warehouses)

# Sum transport costs only for allowed routes (where cost is not None)
transport_fw_cost = gp.quicksum(
    cost_fw[(f, w)] * x_fw[f, w] 
    for f in factories for w in warehouses 
    if cost_fw[(f, w)] is not None
)

transport_wc_cost = gp.quicksum(
    cost_wc[(w, c)] * x_wc[w, c] 
    for w in warehouses for c in customers 
    if cost_wc[(w, c)] is not None
)

model.setObjective(total_fixed_cost + transport_fw_cost + transport_wc_cost, GRB.MINIMIZE)

# 4. Constraints

# (1) Factory Capacity Constraints
for f in factories:
    model.addConstr(gp.quicksum(x_fw[f, w] for w in warehouses) <= factory_capacity[f], name=f"Capacity_{f}")

# (2) Customer Demand Constraints
for c in customers:
    model.addConstr(gp.quicksum(x_wc[w, c] for w in warehouses) == customer_demand[c], name=f"Demand_{c}")

# (3) Flow Conservation (Balance) at Warehouses
for w in warehouses:
    flow_in = gp.quicksum(x_fw[f, w] for f in factories)
    flow_out = gp.quicksum(x_wc[w, c] for c in customers)
    model.addConstr(flow_in == flow_out, name=f"Balance_{w}")

# (4) Prohibited Routes (Where cost is None)
# Factory -> Warehouse
for f in factories:
    for w in warehouses:
        if cost_fw[(f, w)] is None:
            model.addConstr(x_fw[f, w] == 0, name=f"Prohibit_FW_{f}_{w}")

# Warehouse -> Customer
for w in warehouses:
    for c in customers:
        if cost_wc[(w, c)] is None:
            model.addConstr(x_wc[w, c] == 0, name=f"Prohibit_WC_{w}_{c}")

# (5) Warehouse Activation Logic (Indicator Constraints)
# If y[w] == 0, then flow through w must be 0.
# We apply this to the input flow (output flow is forced to 0 by balance constraint).
for w in warehouses:
    flow_in = gp.quicksum(x_fw[f, w] for f in factories)
    # Using addGenConstrIndicator as requested
    model.addGenConstrIndicator(y[w], 0, flow_in == 0, name=f"Indicator_{w}")

# (6) Specific Business Rules
# CityB must be opened (Priority)
model.addConstr(y['CityB'] == 1, name="Priority_CityB")

# At least 2 warehouses must be opened
model.addConstr(gp.quicksum(y[w] for w in warehouses) >= 2, name="Min_Warehouses")

# 5. Solve and Output
model.optimize()

if model.status == GRB.OPTIMAL:
    print("\nOptimal Solution Found:")
    print(f"Objective Value: {model.objVal}")
    for w in warehouses:
        if y[w].x > 0.5:
            print(f"Warehouse {w} Open")
            for f in factories:
                if x_fw[f, w].x > 0:
                    print(f"  <- {x_fw[f, w].x} from {f}")
            for c in customers:
                if x_wc[w, c].x > 0:
                    print(f"  -> {x_wc[w, c].x} to {c}")
    
    # Final Answer Output
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found.")