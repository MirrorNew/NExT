import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and Initialize Model
model = gp.Model("MorningLightSupplyChain")

# 2. Define Parameters and Data Inputs
T = 15
RWs = [1, 2]
RSs = [1, 2, 3, 4]

# Initial Inventory (Period 0)
init_IC = 1200
init_IR = {1: 500, 2: 400}
init_IS = {1: 350, 2: 450, 3: 500, 4: 600}

# Demand per Period
demand = {1: 50, 2: 60, 3: 70, 4: 80}

# Costs
cost_order = 30
cost_hold_wh = 0.2  # Central and Regional
cost_hold_rs = 0.6  # Retail
cost_trans_cw_rw = {1: 0.55, 2: 0.22}
cost_trans_rw_rs = {
    (1, 1): 0.22, (1, 2): 0.20, (1, 3): 0.32, (1, 4): 0.38,
    (2, 1): 0.68, (2, 2): 0.52, (2, 3): 0.34, (2, 4): 0.10
}

# Capacities
cap_cw_rw = 1000
cap_rw_rs = 500

# Constraints Parameters
min_orders_rw1 = 10
safety_stock_ratio = 0.3

# 3. Create Decision Variables
# y[i, t]: Binary order decision for Regional Warehouse i in period t
y = model.addVars(RWs, range(1, T + 1), vtype=GRB.BINARY, name="y")

# z[i, t]: Shipment CW -> RW i in period t (Continuous)
# Upper bound set by capacity
z = model.addVars(RWs, range(1, T + 1), lb=0, ub=cap_cw_rw, vtype=GRB.CONTINUOUS, name="z")

# w[i, j, t]: Shipment RW i -> RS j in period t (Continuous)
# Upper bound set by capacity
w = model.addVars(RWs, RSs, range(1, T + 1), lb=0, ub=cap_rw_rs, vtype=GRB.CONTINUOUS, name="w")

# Inventory Variables (End of period t)
IC = model.addVars(range(1, T + 1), lb=0, vtype=GRB.CONTINUOUS, name="IC")
IR = model.addVars(RWs, range(1, T + 1), lb=0, vtype=GRB.CONTINUOUS, name="IR")
IS = model.addVars(RSs, range(1, T + 1), lb=0, vtype=GRB.CONTINUOUS, name="IS")

# 4. Set up the Objective Function
# Total Cost = Ordering + Holding (CW+RW+RS) + Transportation (CW->RW + RW->RS)
obj_ordering = gp.quicksum(cost_order * y[i, t] for i in RWs for t in range(1, T + 1))
obj_holding_cw = gp.quicksum(cost_hold_wh * IC[t] for t in range(1, T + 1))
obj_holding_rw = gp.quicksum(cost_hold_wh * IR[i, t] for i in RWs for t in range(1, T + 1))
obj_holding_rs = gp.quicksum(cost_hold_rs * IS[j, t] for j in RSs for t in range(1, T + 1))
obj_trans_cw_rw = gp.quicksum(cost_trans_cw_rw[i] * z[i, t] for i in RWs for t in range(1, T + 1))
obj_trans_rw_rs = gp.quicksum(cost_trans_rw_rs[i, j] * w[i, j, t] for i in RWs for j in RSs for t in range(1, T + 1))

model.setObjective(obj_ordering + obj_holding_cw + obj_holding_rw + obj_holding_rs + obj_trans_cw_rw + obj_trans_rw_rs, GRB.MINIMIZE)

# 5. Add Constraints

# 5.1 Indicator Constraints for Ordering
# If y[i, t] == 0 (no order), then z[i, t] == 0 (no shipment)
for i in RWs:
    for t in range(1, T + 1):
        model.addGenConstrIndicator(y[i, t], 0, z[i, t] == 0, name=f"Indicator_Order_{i}_{t}")

# 5.2 Minimum Orders for Regional Warehouse 1
model.addConstr(gp.quicksum(y[1, t] for t in range(1, T + 1)) >= min_orders_rw1, name="MinOrders_RW1")

# 5.3 Safety Stock Requirements for Retail Stores
for j in RSs:
    min_safety_stock = safety_stock_ratio * demand[j]
    for t in range(1, T + 1):
        model.addConstr(IS[j, t] >= min_safety_stock, name=f"SafetyStock_{j}_{t}")

# 5.4 Inventory Flow Balance
# Central Warehouse
# t = 1
model.addConstr(IC[1] == init_IC - gp.quicksum(z[i, 1] for i in RWs), name="Balance_CW_1")
# t > 1
for t in range(2, T + 1):
    model.addConstr(IC[t] == IC[t-1] - gp.quicksum(z[i, t] for i in RWs), name=f"Balance_CW_{t}")

# Regional Warehouses
for i in RWs:
    # t = 1
    model.addConstr(IR[i, 1] == init_IR[i] + z[i, 1] - gp.quicksum(w[i, j, 1] for j in RSs), name=f"Balance_RW_{i}_1")
    # t > 1
    for t in range(2, T + 1):
        model.addConstr(IR[i, t] == IR[i, t-1] + z[i, t] - gp.quicksum(w[i, j, t] for j in RSs), name=f"Balance_RW_{i}_{t}")

# Retail Stores
for j in RSs:
    # t = 1
    model.addConstr(IS[j, 1] == init_IS[j] + gp.quicksum(w[i, j, 1] for i in RWs) - demand[j], name=f"Balance_RS_{j}_1")
    # t > 1
    for t in range(2, T + 1):
        model.addConstr(IS[j, t] == IS[j, t-1] + gp.quicksum(w[i, j, t] for i in RWs) - demand[j], name=f"Balance_RS_{j}_{t}")

# 6. Solve the Model
model.optimize()

# 7. Print Results
if model.Status == GRB.OPTIMAL:
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimization was not successful.")