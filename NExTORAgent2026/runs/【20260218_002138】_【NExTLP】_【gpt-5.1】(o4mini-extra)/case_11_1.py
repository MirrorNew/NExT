import gurobipy as gp
from gurobipy import GRB

# ===============================
# 1. Define all parameter matrices and data inputs
# ===============================

# Parameters List (must use given values)

Factories = ['Shenzhen', 'Vietnam']
CandidateWarehouses = ['CityA', 'CityB', 'CityC']
FixedOperatingCost = {'A': 500, 'B': 400, 'C': 300}
CostReductionTarget = 0.15  # not used numerically because C0 is unknown
CustomerRegions = ['Singapore', 'Malaysia', 'Philippines']

AllowedRoutes_FactoryWarehouse = [
    ['Shenzhen', 'CityA'],
    ['Shenzhen', 'CityB'],
    ['Shenzhen', 'CityC'],
    ['Vietnam', 'CityB'],
    ['Vietnam', 'CityC'],
]

AllowedRoutes_WarehouseCustomer = [
    ['CityA', 'Singapore'],
    ['CityA', 'Malaysia'],
    ['CityB', 'Singapore'],
    ['CityB', 'Malaysia'],
    ['CityB', 'Philippines'],
    ['CityC', 'Singapore'],
    ['CityC', 'Malaysia'],
    ['CityC', 'Philippines'],
]

PriorityWarehouse = ['CityB']
MinWarehousesOpen = 2

Table_1 = {
    'ShenzhenFactory': 1000,
    'VietnamFactory': 800,
    'Customer1': 500,
    'Customer2': 700,
    'Customer3': 500
}

Table_2 = {
    'CityA': {'Customer1': 3, 'Customer2': 4, 'Customer3': None},
    'CityB': {'Customer1': None, 'Customer2': 3, 'Customer3': 3},
    'CityC': {'Customer1': 3, 'Customer2': 5, 'Customer3': 2}
}

Table_3 = {
    'CityA': {'Shenzhen': 2, 'Vietnam': None},
    'CityB': {'Shenzhen': 4, 'Vietnam': 1},
    'CityC': {'Shenzhen': 3, 'Vietnam': 2}
}

# Map to model indices used in context
factories = ['S', 'V']  # Shenzhen, Vietnam
warehouses = ['A', 'B', 'C']  # CityA, CityB, CityC
customers = [1, 2, 3]  # Customer1, Customer2, Customer3

# Capacities and demands
cap_S = Table_1['ShenzhenFactory']
cap_V = Table_1['VietnamFactory']
demand = {
    1: Table_1['Customer1'],
    2: Table_1['Customer2'],
    3: Table_1['Customer3']
}

# Fixed costs by warehouse index
fixed_cost = {
    'A': FixedOperatingCost['A'],
    'B': FixedOperatingCost['B'],
    'C': FixedOperatingCost['C']
}

# Factory -> Warehouse transportation cost
c_FW = {}
# Shenzhen
c_FW[('S', 'A')] = Table_3['CityA']['Shenzhen']
c_FW[('S', 'B')] = Table_3['CityB']['Shenzhen']
c_FW[('S', 'C')] = Table_3['CityC']['Shenzhen']
# Vietnam
# Vietnam -> CityA is forbidden (None), cost not defined
c_FW[('V', 'B')] = Table_3['CityB']['Vietnam']
c_FW[('V', 'C')] = Table_3['CityC']['Vietnam']

# Warehouse -> Customer transportation cost
c_WC = {}
# Map CityA,B,C & Customer1,2,3 to A,B,C & 1,2,3
c_WC[('A', 1)] = Table_2['CityA']['Customer1']
c_WC[('A', 2)] = Table_2['CityA']['Customer2']
# Customer3 from CityA is forbidden
c_WC[('B', 2)] = Table_2['CityB']['Customer2']
c_WC[('B', 3)] = Table_2['CityB']['Customer3']
c_WC[('C', 1)] = Table_2['CityC']['Customer1']
c_WC[('C', 2)] = Table_2['CityC']['Customer2']
c_WC[('C', 3)] = Table_2['CityC']['Customer3']

# Big-M for indicator constraints
M = cap_S + cap_V  # 1800

# ===============================
# 2. Create model
# ===============================
model = gp.Model("SupplyChainNetworkDesign")

# ===============================
# 3. Create decision variables
# ===============================

# Binary variables: warehouse open
y = model.addVars(warehouses, vtype=GRB.BINARY, name="y")

# Flows: factory -> warehouse
x_FW_vars = {}
for i in factories:
    for k in warehouses:
        # Check if route is allowed based on AllowedRoutes_FactoryWarehouse
        # Map indices back to given names
        if i == 'S':
            fac_name = 'Shenzhen'
        else:
            fac_name = 'Vietnam'
        if k == 'A':
            city_name = 'CityA'
        elif k == 'B':
            city_name = 'CityB'
        else:
            city_name = 'CityC'

        if [fac_name, city_name] in AllowedRoutes_FactoryWarehouse:
            x_FW_vars[(i, k)] = model.addVar(lb=0.0, ub=M, vtype=GRB.CONTINUOUS,
                                             name=f"x_{i}_{k}")
        else:
            # Forbidden: fix to 0
            x_FW_vars[(i, k)] = model.addVar(lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS,
                                             name=f"x_{i}_{k}_forbidden")

# Flows: warehouse -> customer
x_WC_vars = {}
for k in warehouses:
    for j in customers:
        # Map to parameter names
        if k == 'A':
            city_name = 'CityA'
        elif k == 'B':
            city_name = 'CityB'
        else:
            city_name = 'CityC'

        cust_name = f"Customer{j}"
        # Translate to AllowedRoutes_WarehouseCustomer
        if j == 1:
            region_name = 'Singapore'
        elif j == 2:
            region_name = 'Malaysia'
        else:
            region_name = 'Philippines'

        if [city_name, region_name] in AllowedRoutes_WarehouseCustomer and Table_2[city_name][cust_name] is not None:
            x_WC_vars[(k, j)] = model.addVar(lb=0.0, ub=M, vtype=GRB.CONTINUOUS,
                                             name=f"x_{k}_{j}")
        else:
            # Forbidden: fix to 0
            x_WC_vars[(k, j)] = model.addVar(lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS,
                                             name=f"x_{k}_{j}_forbidden")

model.update()

# ===============================
# 4. Set up the objective function
# ===============================

obj = gp.LinExpr()

# Fixed warehouse costs
for k in warehouses:
    obj += fixed_cost[k] * y[k]

# Factory -> warehouse costs
for (i, k), var in x_FW_vars.items():
    if (i, k) in c_FW and c_FW[(i, k)] is not None:
        obj += c_FW[(i, k)] * var

# Warehouse -> customer costs
for (k, j), var in x_WC_vars.items():
    if (k, j) in c_WC and c_WC[(k, j)] is not None:
        obj += c_WC[(k, j)] * var

model.setObjective(obj, GRB.MINIMIZE)

# ===============================
# 5. Add constraints
# ===============================

# Demand satisfaction: sum_k x_{k,j} = d_j
for j in customers:
    model.addConstr(
        gp.quicksum(x_WC_vars[(k, j)] for k in warehouses) == demand[j],
        name=f"Demand_{j}"
    )

# Factory capacities
model.addConstr(
    x_FW_vars[('S', 'A')] + x_FW_vars[('S', 'B')] + x_FW_vars[('S', 'C')] <= cap_S,
    name="Cap_Shenzhen"
)

model.addConstr(
    x_FW_vars[('V', 'A')] + x_FW_vars[('V', 'B')] + x_FW_vars[('V', 'C')] <= cap_V,
    name="Cap_Vietnam"
)

# Flow balance at warehouses: in = out
for k in warehouses:
    inbound = gp.quicksum(x_FW_vars[(i, k)] for i in factories)
    outbound = gp.quicksum(x_WC_vars[(k, j)] for j in customers)
    model.addConstr(inbound == outbound, name=f"FlowBalance_{k}")

# Warehouse activation coupling using indicator constraints
# If y_k = 0, all flows to/from k must be 0
# We also ensure feasibility for y_k = 1 by allowing nonnegative flows (already ensured by bounds)

for k in warehouses:
    # Inbound flows
    for i in factories:
        var = x_FW_vars[(i, k)]
        # y_k = 0 -> var <= 0
        model.addGenConstrIndicator(y[k], 0, var <= 0, name=f"Ind_in_{i}_{k}_off")
        # y_k = 1 -> var >= 0 (already lb=0, but add per instruction)
        model.addGenConstrIndicator(y[k], 1, var >= 0, name=f"Ind_in_{i}_{k}_on")

    # Outbound flows
    for j in customers:
        var = x_WC_vars[(k, j)]
        # y_k = 0 -> var <= 0
        model.addGenConstrIndicator(y[k], 0, var <= 0, name=f"Ind_out_{k}_{j}_off")
        # y_k = 1 -> var >= 0
        model.addGenConstrIndicator(y[k], 1, var >= 0, name=f"Ind_out_{k}_{j}_on")

# Priority: open warehouse B
model.addConstr(y['B'] == 1, name="Open_B")

# Minimum number of warehouses
model.addConstr(y['A'] + y['B'] + y['C'] >= MinWarehousesOpen, name="MinWarehouses")

# Note: Cost reduction constraint with unknown C0 is not explicitly enforced.
# The objective already minimizes cost; FinalAnswer will be that minimum total cost.

# ===============================
# 6. Solve the model
# ===============================

model.setParam('OutputFlag', 0)  # Silent mode
model.optimize()

# ===============================
# 7. Print results and FinalAnswer
# ===============================

if model.Status == GRB.OPTIMAL:
    total_cost = model.ObjVal
else:
    total_cost = None

# Print some basic results (optional)
if total_cost is not None:
    print(f"Optimal total cost: {total_cost}")
    for k in warehouses:
        print(f"Warehouse {k} open: {int(round(y[k].X))}")
else:
    print("No optimal solution found.")

# Final answer required by the problem: total minimal supply chain cost
print(f"FinalAnswer=【{total_cost}】")