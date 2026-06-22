import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
factories = ['Shenzhen', 'Vietnam']
warehouses = ['CityA', 'CityB', 'CityC']
customers = ['Customer1', 'Customer2', 'Customer3']

# Factory Capacities
factory_cap = {
    'Shenzhen': 1000,
    'Vietnam': 800
}

# Customer Demands
customer_dem = {
    'Customer1': 500,
    'Customer2': 700,
    'Customer3': 500
}

# Warehouse Fixed Operating Costs
fixed_costs = {
    'CityA': 500,
    'CityB': 400,
    'CityC': 300
}

# Unit Transportation Costs from Factory to Warehouse
# According to Table 3 and route restrictions
f_w_cost = {
    ('Shenzhen', 'CityA'): 2,
    ('Shenzhen', 'CityB'): 4,
    ('Shenzhen', 'CityC'): 3,
    ('Vietnam', 'CityB'): 1,
    ('Vietnam', 'CityC'): 2
}

# Unit Transportation Costs from Warehouse to Customer
# According to Table 2 and waterway control restrictions
w_c_cost = {
    ('CityA', 'Customer1'): 3,
    ('CityA', 'Customer2'): 4,
    # ('CityA', 'Customer3'): None (Restricted)
    # ('CityB', 'Customer1'): None (Restricted)
    ('CityB', 'Customer2'): 3,
    ('CityB', 'Customer3'): 3,
    ('CityC', 'Customer1'): 3,
    ('CityC', 'Customer2'): 5,
    ('CityC', 'Customer3'): 2
}

# 2. Create Gurobi Model
model = gp.Model("Huanya_Manufacturing_Optimization")

# 3. Decision Variables
# Binary variables for warehouse activation
y = model.addVars(warehouses, vtype=GRB.BINARY, name="y")

# Continuous variables for transportation flow
x_fw = model.addVars(f_w_cost.keys(), vtype=GRB.CONTINUOUS, name="x_fw")
x_wc = model.addVars(w_c_cost.keys(), vtype=GRB.CONTINUOUS, name="x_wc")

# 4. Set up the Objective Function
# Objective: Minimize (Fixed Costs + Factory-to-Warehouse Costs + Warehouse-to-Customer Costs)
total_fixed_cost = gp.quicksum(fixed_costs[w] * y[w] for w in warehouses)
total_fw_cost = gp.quicksum(f_w_cost[f, w] * x_fw[f, w] for f, w in f_w_cost.keys())
total_wc_cost = gp.quicksum(w_c_cost[w, c] * x_wc[w, c] for w, c in w_c_cost.keys())
model.setObjective(total_fixed_cost + total_fw_cost + total_wc_cost, GRB.MINIMIZE)

# 5. Add all constraints
# Factory Capacity Constraints
for f in factories:
    model.addConstr(
        gp.quicksum(x_fw[f, w] for w_inner in warehouses if (f, w_inner) in f_w_cost) <= factory_cap[f],
        name=f"Cap_{f}"
    )

# Customer Demand Satisfaction Constraints
for c in customers:
    model.addConstr(
        gp.quicksum(x_wc[w, c] for w in warehouses if (w, c) in w_c_cost) == customer_dem[c],
        name=f"Dem_{c}"
    )

# Warehouse Flow Balance and Indicator Constraints
for w in warehouses:
    # Flow balance: Sum of inflow from factories must equal sum of outflow to customers
    inbound = gp.quicksum(x_fw[f, w_inner] for f, w_inner in f_w_cost.keys() if w_inner == w)
    outbound = gp.quicksum(x_wc[w_inner, c] for w_inner, c in w_c_cost.keys() if w_inner == w)
    model.addConstr(inbound == outbound, name=f"Balance_{w}")
    
    # Indicator Constraints: If warehouse is not activated (y=0), flows must be zero.
    # The sum of inbound/outbound is used as a proxy for all individual routes.
    model.addGenConstrIndicator(y[w], 0, inbound == 0, name=f"Inbound_Inactive_{w}")
    model.addGenConstrIndicator(y[w], 0, outbound == 0, name=f"Outbound_Inactive_{w}")

# Warehouse Strategy Constraints
# 1. City B must be opened (policy priority/subsidies)
model.addConstr(y['CityB'] == 1, name="Priority_B")

# 2. At least 2 warehouses must be opened
model.addConstr(gp.quicksum(y[w] for w in warehouses) >= 2, name="Min_Warehouses")

# 6. Solve the model and print results
model.optimize()

if model.status == GRB.OPTIMAL:
    # The question asks for the minimum total cost of the supply chain.
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found.")