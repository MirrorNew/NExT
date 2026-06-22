import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
# Fixed costs for activating warehouses
fixed_costs = {'CityA': 500, 'CityB': 400, 'CityC': 300}

# Factory capacities and Customer demands
factory_cap = {'Shenzhen': 1000, 'Vietnam': 800}
customer_dem = {'Customer1': 500, 'Customer2': 700, 'Customer3': 500}

# Unit Transportation Costs from Factory to Warehouse (Table 3)
# Shenzhen can ship to A, B, C. Vietnam can only ship to B, C.
f_w_costs = {
    ('Shenzhen', 'CityA'): 2,
    ('Shenzhen', 'CityB'): 4,
    ('Shenzhen', 'CityC'): 3,
    ('Vietnam', 'CityB'): 1,
    ('Vietnam', 'CityC'): 2
}

# Unit Transportation Costs from Warehouse to Customer (Table 2)
# CityA serves Singapore(1), Malaysia(2).
# CityB serves Malaysia(2), Philippines(3).
# CityC serves all three.
w_c_costs = {
    ('CityA', 'Customer1'): 3,
    ('CityA', 'Customer2'): 4,
    ('CityB', 'Customer2'): 3,
    ('CityB', 'Customer3'): 3,
    ('CityC', 'Customer1'): 3,
    ('CityC', 'Customer2'): 5,
    ('CityC', 'Customer3'): 2
}

# 2. Create the Gurobi model
model = gp.Model("Huanya_Manufacturing_Optimization")

# 3. Decision Variables
# Binary variables for warehouse activation
y = model.addVars(fixed_costs.keys(), vtype=GRB.BINARY, name="y")

# Continuous variables for transportation flow
x_fw = model.addVars(f_w_costs.keys(), vtype=GRB.CONTINUOUS, name="x_fw")
x_wc = model.addVars(w_c_costs.keys(), vtype=GRB.CONTINUOUS, name="x_wc")

# 4. Set up the Objective Function
# Goal is to minimize total supply chain cost (Fixed costs + transportation costs)
total_fixed_cost = gp.quicksum(fixed_costs[k] * y[k] for k in fixed_costs)
total_fw_cost = gp.quicksum(f_w_costs[f, w] * x_fw[f, w] for f, w in f_w_costs)
total_wc_cost = gp.quicksum(w_c_costs[w, c] * x_wc[w, c] for w, c in w_c_costs)
model.setObjective(total_fixed_cost + total_fw_cost + total_wc_cost, GRB.MINIMIZE)

# 5. Add all constraints
# Factory Capacity Constraints
for f in factory_cap:
    model.addConstr(gp.quicksum(x_fw[fi, wi] for fi, wi in f_w_costs if fi == f) <= factory_cap[f], name=f"Cap_{f}")

# Customer Demand Satisfaction Constraints
for c in customer_dem:
    model.addConstr(gp.quicksum(x_wc[wi, ci] for wi, ci in w_c_costs if ci == c) == customer_dem[c], name=f"Dem_{c}")

# Warehouse Flow Balance and Indicator Constraints
for w in fixed_costs:
    # Inflow to warehouse must equal outflow
    inbound = gp.quicksum(x_fw[fi, wi] for fi, wi in f_w_costs if wi == w)
    outbound = gp.quicksum(x_wc[wi, ci] for wi, ci in w_c_costs if wi == w)
    model.addConstr(inbound == outbound, name=f"Balance_{w}")
    
    # Activation requirement: If y[w] = 0, no flow is allowed through the warehouse
    model.addGenConstrIndicator(y[w], 0, inbound == 0, name=f"Ind_In_{w}")
    model.addGenConstrIndicator(y[w], 0, outbound == 0, name=f"Ind_Out_{w}")

# Warehouse Priority and Strategic Constraints
# City B must be activated (Priority subsidy)
model.addConstr(y['CityB'] == 1, name="Priority_B")

# At least 2 warehouses must be opened
model.addConstr(gp.quicksum(y[k] for k in fixed_costs) >= 2, name="Min_Warehouse_Requirement")

# 6. Solve the model and print results
model.optimize()

if model.status == GRB.OPTIMAL:
    # The output requires the minimum supply chain cost obtained.
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("Optimal solution not found.")