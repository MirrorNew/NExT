import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters (from Parameters List, values used as-is)
# =========================

Plants = ['A1', 'A2']
Warehouses_existing = ['B1', 'B2', 'B3', 'B4']
Warehouses_new = ['B5', 'B6']
Warehouses_all = Warehouses_existing + Warehouses_new
Customers = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']

Plant_Capacity = [150000.0, 200000.0]        # [A1, A2]
Warehouse_Capacity_Base = [70000.0, 50000.0, 100000.0, 40000.0]  # [B1,B2,B3,B4]
Warehouse_Capacity_Expansion_Increment = 20000.0
New_Warehouse_Capacity = [30000.0, 25000.0]  # [B5,B6]
Demand = [50000.0, 10000.0, 40000.0, 35000.0, 60000.0, 20000.0]  # [C1..C6]
Max_Open_Warehouses = 4

Build_Cost_New_Warehouses = [1200000.0, 400000.0]  # [B5,B6]
Expansion_Cost_B2 = 300000.0
Closure_Savings = [100000.0, 50000.0]  # [B1,B4]

Transport_Cost_Plant_to_Warehouse = [
    [50.0, 50.0, 100.0, 20.0, None, None],   # A1 to [B1,B2,B3,B4,B5,B6]
    [None, 30.0, 50.0, 20.0, None, None]     # A2 to [B1,B2,B3,B4,B5,B6]
]

Transport_Cost_Plant_to_Customer = [
    [100.0, None, 150.0, 200.0, None, 100.0],  # A1 to C1..C6
    [200.0, None, None, None, None, None]      # A2 to C1..C6
]

Transport_Cost_Warehouse_to_Customer_Existing = [
    [None, 150.0, 50.0, 150.0, None, 100.0],   # B1 to C1..C6
    [None, None, 100.0, 50.0, 50.0, 100.0],    # B2
    [None, None, None, None, None, None],      # B3 (we will add C2,C3,C5,C6 via model context)
    [None, None, 150.0, 200.0, None, 50.0]     # B4 (we will add C1,C2,C3,C4 via model context)
]

Transport_Cost_New_Warehouse_Arcs = [
    {'From': 'A1', 'To': 'B5', 'Cost': 60.0},
    {'From': 'A1', 'To': 'B6', 'Cost': 40.0},
    {'From': 'A2', 'To': 'B5', 'Cost': 40.0},
    {'From': 'A2', 'To': 'B6', 'Cost': 30.0},
    {'From': 'B5', 'To': 'C1', 'Cost': 120.0},
    {'From': 'B5', 'To': 'C2', 'Cost': 60.0},
    {'From': 'B5', 'To': 'C3', 'Cost': 40.0},
    {'From': 'B5', 'To': 'C5', 'Cost': 30.0},
    {'From': 'B5', 'To': 'C6', 'Cost': 80.0},
    {'From': 'B6', 'To': 'C2', 'Cost': 40.0},
    {'From': 'B6', 'To': 'C4', 'Cost': 50.0},
    {'From': 'B6', 'To': 'C5', 'Cost': 60.0},
    {'From': 'B6', 'To': 'C6', 'Cost': 90.0}
]

Preference_C1_from_A1_Min = 50000.0
Preference_C2_from_B1_Min = 10000.0
Preference_C5_from_B2_Min = 60000.0
Preference_C6_from_B3_or_B4_Min = 20000.0

Big_M = 215000.0

# =========================
# 2. Derived mappings
# =========================

plant_cap = {Plants[i]: Plant_Capacity[i] for i in range(len(Plants))}
wh_cap_base = {Warehouses_existing[i]: Warehouse_Capacity_Base[i] for i in range(len(Warehouses_existing))}
wh_cap_new = {Warehouses_new[i]: New_Warehouse_Capacity[i] for i in range(len(Warehouses_new))}
demand = {Customers[i]: Demand[i] for i in range(len(Customers))}

# Cost dictionaries
cost_plant_wh = {}
# existing warehouses part
for i, p in enumerate(Plants):
    for j, w in enumerate(Warehouses_all):
        c = Transport_Cost_Plant_to_Warehouse[i][j] if j < len(Transport_Cost_Plant_to_Warehouse[i]) else None
        if c is not None:
            cost_plant_wh[(p, w)] = c

# add explicit plant->new warehouse from Transport_Cost_New_Warehouse_Arcs
for arc in Transport_Cost_New_Warehouse_Arcs:
    f, t, c = arc['From'], arc['To'], arc['Cost']
    if f in Plants and t in Warehouses_new:
        cost_plant_wh[(f, t)] = c

cost_plant_cust = {}
for i, p in enumerate(Plants):
    for k, c in enumerate(Transport_Cost_Plant_to_Customer[i]):
        if c is not None:
            cost_plant_cust[(p, Customers[k])] = c

cost_wh_cust = {}
# existing warehouses base costs
for j, w in enumerate(Warehouses_existing):
    for k, c in enumerate(Transport_Cost_Warehouse_to_Customer_Existing[j]):
        if c is not None:
            cost_wh_cust[(w, Customers[k])] = c

# context specifies further arcs for B3 and B4:
# B3: C2=150, C3=200, C5=50, C6=150
cost_wh_cust[('B3', 'C2')] = 150.0
cost_wh_cust[('B3', 'C3')] = 200.0
cost_wh_cust[('B3', 'C5')] = 50.0
cost_wh_cust[('B3', 'C6')] = 150.0
# B4: C1=100, C2=50, C3=50, C4=100
cost_wh_cust[('B4', 'C1')] = 100.0
cost_wh_cust[('B4', 'C2')] = 50.0
cost_wh_cust[('B4', 'C3')] = 50.0
cost_wh_cust[('B4', 'C4')] = 100.0

# new warehouses -> customers
for arc in Transport_Cost_New_Warehouse_Arcs:
    f, t, c = arc['From'], arc['To'], arc['Cost']
    if f in Warehouses_new and t in Customers:
        cost_wh_cust[(f, t)] = c

# =========================
# 3. Create model
# =========================

model = gp.Model("Warehouse_Expansion_And_Selection")

# =========================
# 4. Decision variables
# =========================

# flows: plant -> warehouse
x_pw = model.addVars(cost_plant_wh.keys(), lb=0.0, vtype=GRB.CONTINUOUS, name="x_pw")
# flows: plant -> customer
x_pc = model.addVars(cost_plant_cust.keys(), lb=0.0, vtype=GRB.CONTINUOUS, name="x_pc")
# flows: warehouse -> customer
x_wc = model.addVars(cost_wh_cust.keys(), lb=0.0, vtype=GRB.CONTINUOUS, name="x_wc")

# binary facility decisions
y_5 = model.addVar(vtype=GRB.BINARY, name="y_5")        # build B5
y_6 = model.addVar(vtype=GRB.BINARY, name="y_6")        # build B6
y2_exp = model.addVar(vtype=GRB.BINARY, name="y2_exp")  # expand B2
z_1 = model.addVar(vtype=GRB.BINARY, name="z_1")        # close B1
z_4 = model.addVar(vtype=GRB.BINARY, name="z_4")        # close B4

# logical open indicators
w_1 = model.addVar(vtype=GRB.BINARY, name="w_1")
w_2 = model.addVar(vtype=GRB.BINARY, name="w_2")
w_3 = model.addVar(vtype=GRB.BINARY, name="w_3")
w_4 = model.addVar(vtype=GRB.BINARY, name="w_4")
w_5 = model.addVar(vtype=GRB.BINARY, name="w_5")
w_6 = model.addVar(vtype=GRB.BINARY, name="w_6")

# relations for logical indicators
model.addConstr(w_1 + z_1 == 1, name="rel_w1_z1")
model.addConstr(w_4 + z_4 == 1, name="rel_w4_z4")
model.addConstr(w_2 == 1, name="rel_w2")
model.addConstr(w_3 == 1, name="rel_w3")
model.addConstr(w_5 == y_5, name="rel_w5_y5")
model.addConstr(w_6 == y_6, name="rel_w6_y6")

# =========================
# 5. Objective function
# =========================

transport_cost = (
    gp.quicksum(cost_plant_wh[a] * x_pw[a] for a in cost_plant_wh) +
    gp.quicksum(cost_plant_cust[a] * x_pc[a] for a in cost_plant_cust) +
    gp.quicksum(cost_wh_cust[a] * x_wc[a] for a in cost_wh_cust)
)

fixed_cost = (
    Build_Cost_New_Warehouses[0] * y_5 +
    Build_Cost_New_Warehouses[1] * y_6 +
    Expansion_Cost_B2 * y2_exp -
    Closure_Savings[0] * z_1 -
    Closure_Savings[1] * z_4
)

model.setObjective(transport_cost + fixed_cost, GRB.MINIMIZE)

# =========================
# 6. Constraints
# =========================

# helper functions
def inflow_to_wh(w):
    return [key for key in x_pw.keys() if key[1] == w]

def outflow_from_wh(w):
    return [key for key in x_wc.keys() if key[0] == w]

# 6.1 Plant capacity constraints
for p in Plants:
    out_pw = [key for key in x_pw.keys() if key[0] == p]
    out_pc = [key for key in x_pc.keys() if key[0] == p]
    model.addConstr(
        gp.quicksum(x_pw[a] for a in out_pw) +
        gp.quicksum(x_pc[a] for a in out_pc) <= plant_cap[p],
        name=f"Plant_cap_{p}"
    )

# 6.2 Customer demand constraints
for c in Customers:
    incoming_pc = [key for key in x_pc.keys() if key[1] == c]
    incoming_wc = [key for key in x_wc.keys() if key[1] == c]
    model.addConstr(
        gp.quicksum(x_pc[a] for a in incoming_pc) +
        gp.quicksum(x_wc[a] for a in incoming_wc) >= demand[c],
        name=f"Demand_{c}"
    )

# 6.3 Preference constraints
if ('A1', 'C1') in x_pc:
    model.addConstr(x_pc[('A1', 'C1')] >= Preference_C1_from_A1_Min,
                    name="Pref_C1_from_A1")

if ('B1', 'C2') in x_wc:
    model.addConstr(x_wc[('B1', 'C2')] >= Preference_C2_from_B1_Min,
                    name="Pref_C2_from_B1")

if ('B2', 'C5') in x_wc:
    model.addConstr(x_wc[('B2', 'C5')] >= Preference_C5_from_B2_Min,
                    name="Pref_C5_from_B2")

terms_C6_pref = []
if ('B3', 'C6') in x_wc:
    terms_C6_pref.append(x_wc[('B3', 'C6')])
if ('B4', 'C6') in x_wc:
    terms_C6_pref.append(x_wc[('B4', 'C6')])
if terms_C6_pref:
    model.addConstr(gp.quicksum(terms_C6_pref) >= Preference_C6_from_B3_or_B4_Min,
                    name="Pref_C6_from_B3_or_B4")

# 6.4 Warehouse capacity constraints
# B1: sum_k x_{B1,Ck} <= 70000*(1-z_1)
if 'B1' in Warehouses_existing:
    out_B1 = outflow_from_wh('B1')
    model.addConstr(
        gp.quicksum(x_wc[a] for a in out_B1) <= wh_cap_base['B1'] * (1 - z_1),
        name="Warehouse_B1_capacity"
    )

# B2: sum_k x_{B2,Ck} <= 50000 + 20000*y2_exp
if 'B2' in Warehouses_existing:
    out_B2 = outflow_from_wh('B2')
    model.addConstr(
        gp.quicksum(x_wc[a] for a in out_B2) <= wh_cap_base['B2'] +
        Warehouse_Capacity_Expansion_Increment * y2_exp,
        name="Warehouse_B2_capacity_with_expansion"
    )

# B3: sum_k x_{B3,Ck} <= 100000
if 'B3' in Warehouses_existing:
    out_B3 = outflow_from_wh('B3')
    model.addConstr(
        gp.quicksum(x_wc[a] for a in out_B3) <= wh_cap_base['B3'],
        name="Warehouse_B3_capacity"
    )

# B4: sum_k x_{B4,Ck} <= 40000*(1-z_4)
if 'B4' in Warehouses_existing:
    out_B4 = outflow_from_wh('B4')
    model.addConstr(
        gp.quicksum(x_wc[a] for a in out_B4) <= wh_cap_base['B4'] * (1 - z_4),
        name="Warehouse_B4_capacity"
    )

# B5: sum_k x_{B5,Ck} <= 30000*y_5
if 'B5' in Warehouses_new:
    out_B5 = outflow_from_wh('B5')
    model.addConstr(
        gp.quicksum(x_wc[a] for a in out_B5) <= wh_cap_new['B5'] * y_5,
        name="Warehouse_B5_capacity"
    )

# B6: sum_k x_{B6,Ck} <= 25000*y_6
if 'B6' in Warehouses_new:
    out_B6 = outflow_from_wh('B6')
    model.addConstr(
        gp.quicksum(x_wc[a] for a in out_B6) <= wh_cap_new['B6'] * y_6,
        name="Warehouse_B6_capacity"
    )

# 6.5 Warehouse flow balance: inflow = outflow
for w in Warehouses_all:
    inflow = gp.quicksum(x_pw[a] for a in inflow_to_wh(w))
    outflow = gp.quicksum(x_wc[a] for a in outflow_from_wh(w))
    model.addConstr(inflow == outflow, name=f"Flow_balance_{w}")

# 6.6 Max 4 warehouses operating:
# (1 - z_1) + 1 + 1 + (1 - z_4) + y_5 + y_6 <= 4
model.addConstr(
    (1 - z_1) + 1 + 1 + (1 - z_4) + y_5 + y_6 <= Max_Open_Warehouses,
    name="Max_4_warehouses_operating"
)

# 6.7 Big-M constraints: no flow if closed/unbuilt
for c in Customers:
    if ('B1', c) in x_wc:
        model.addConstr(
            x_wc[('B1', c)] <= Big_M * (1 - z_1),
            name=f"No_flow_through_closed_B1_{c}"
        )
    if ('B4', c) in x_wc:
        model.addConstr(
            x_wc[('B4', c)] <= Big_M * (1 - z_4),
            name=f"No_flow_through_closed_B4_{c}"
        )
    if ('B5', c) in x_wc:
        model.addConstr(
            x_wc[('B5', c)] <= Big_M * y_5,
            name=f"No_flow_through_unbuilt_B5_{c}"
        )
    if ('B6', c) in x_wc:
        model.addConstr(
            x_wc[('B6', c)] <= Big_M * y_6,
            name=f"No_flow_through_unbuilt_B6_{c}"
        )

# 6.8 Nonnegativity is enforced by lb=0.0 on all flow variables

# =========================
# 7. Solve model
# =========================

model.Params.OutputFlag = 1
model.optimize()

# =========================
# 8. Print results and FinalAnswer
# =========================

if model.Status == GRB.OPTIMAL:
    total_cost = model.ObjVal
    print("Optimal solution found.")
    print(f"Total cost = {total_cost:.2f}")

    print("\nFacility decisions:")
    print(f"B5 built (y_5) = {y_5.X}")
    print(f"B6 built (y_6) = {y_6.X}")
    print(f"B2 expanded (y2_exp) = {y2_exp.X}")
    print(f"B1 closed (z_1) = {z_1.X}")
    print(f"B4 closed (z_4) = {z_4.X}")

    print("\nOpen indicators:")
    print(f"B1 open (w_1) = {w_1.X}")
    print(f"B2 open (w_2) = {w_2.X}")
    print(f"B3 open (w_3) = {w_3.X}")
    print(f"B4 open (w_4) = {w_4.X}")
    print(f"B5 open (w_5) = {w_5.X}")
    print(f"B6 open (w_6) = {w_6.X}")

    print("\nFlows Plant -> Warehouse:")
    for (p, w), var in x_pw.items():
        if var.X > 1e-6:
            print(f"x_{p}_{w} = {var.X:.2f}")

    print("\nFlows Plant -> Customer:")
    for (p, c), var in x_pc.items():
        if var.X > 1e-6:
            print(f"x_{p}_{c} = {var.X:.2f}")

    print("\nFlows Warehouse -> Customer:")
    for (w, c), var in x_wc.items():
        if var.X > 1e-6:
            print(f"x_{w}_{c} = {var.X:.2f}")

    # FinalAnswer: the minimal total cost
    print(f"FinalAnswer=【{total_cost:.2f}】")
else:
    print(f"Model not optimal. Status = {model.Status}")
    print("FinalAnswer=【None】")