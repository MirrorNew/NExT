import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Import / Parameters
# =========================

# Use exactly the given Parameters List values

I = [1, 2]
J_orig = [1, 2, 3, 4]
K = [1, 2, 3, 4, 5, 6]

PreAssignment = [
    [],                 # index 0 dummy
    [['A', 1]],         # k=1: C1 -> A1
    [['B', 1]],         # k=2: C2 -> B1
    [],                 # k=3: no special
    [],                 # k=4: no special
    [['B', 2]],         # k=5: C5 -> B2
    [['B', 3], ['B', 4]]# k=6: C6 -> B3 or B4
]

MaxSupply = [None, 150000, 200000]      # plant capacities
MaxTurnover = [None, 70000, 50000, 100000, 40000]  # B1..B4 base
MinDemand = [None, 50000, 10000, 40000, 35000, 60000, 20000]  # C1..C6

NewWarehousesCandidates = [5, 6]
ExpandCandidate = 2
MaxOpenWarehouses = 4
ClosingCandidates = [1, 4]

Inv_B5 = 1200000
Cap_B5 = 30000
Inv_B6 = 400000
Cap_B6 = 25000
Inv_ExpB2 = 300000
Cap_ExpB2 = 20000
Save_B1 = 100000
Save_B4 = 50000

Table_1_C16 = [
    ['A1', [50, 50, 100, 20, 100, None, 150, 200, None, 100]],
    ['A2', [None, 30, 50, 20, 200, None, None, None, None, None]],
    ['B1', [None, None, None, None, None, 150, 50, 150, None, 100]],
    [None, [None, None, 100, 50, 50, 100, 50, None, None]],
    ['B3', [None, None, None, None, None, 150, 200, None, 50, 150]]
]
Table_2_C17 = [
    ['A1', [60, 40, None, None, None, None, None, None]],
    ['A2', [40, 30, None, None, None, None, None, None]],
    ['B5', [None, None, 120, 60, 40, None, 30, 80]],
    ['B6', [None, None, None, 40, None, 50, 60, 90]]
]

# All potential warehouses
J = [1, 2, 3, 4, 5, 6]

# =========================
# 2. Build cost matrices
# =========================

# Cost from plants to warehouses c^{orig}_{i,j}
cAW = {i: {j: None for j in J} for i in I}

A1_row = Table_1_C16[0][1]  # A1 row C16
A2_row = Table_1_C16[1][1]  # A2 row C16

# A1 -> B1..B4
cAW[1][1] = A1_row[0]
cAW[1][2] = A1_row[1]
cAW[1][3] = A1_row[2]
cAW[1][4] = A1_row[3]

# A2 -> B1..B4
cAW[2][1] = A2_row[0]
cAW[2][2] = A2_row[1]
cAW[2][3] = A2_row[2]
cAW[2][4] = A2_row[3]

# From C17: A1/A2 -> B5,B6
A1_row_17 = Table_2_C17[0][1]
A2_row_17 = Table_2_C17[1][1]
cAW[1][5] = A1_row_17[0]
cAW[1][6] = A1_row_17[1]
cAW[2][5] = A2_row_17[0]
cAW[2][6] = A2_row_17[1]

# Cost from plants directly to customers c^{dir}_{i,k}
cAC = {i: {k: None for k in K} for i in I}
# From C16 A1,C1..C6
cAC[1][1] = A1_row[4]
cAC[1][2] = A1_row[5]
cAC[1][3] = A1_row[6]
cAC[1][4] = A1_row[7]
cAC[1][5] = A1_row[8]
cAC[1][6] = A1_row[9]
# From C16 A2,C1..C6
cAC[2][1] = A2_row[4]
cAC[2][2] = A2_row[5]
cAC[2][3] = A2_row[6]
cAC[2][4] = A2_row[7]
cAC[2][5] = A2_row[8]
cAC[2][6] = A2_row[9]

# Cost from warehouses to customers c^{wh}_{j,k}
cWC = {j: {k: None for k in K} for j in J}

B1_row = Table_1_C16[2][1]
B2_row = Table_1_C16[3][1]
B3_row = Table_1_C16[4][1]

# Map B1 -> C1..C6 using positions 4..9
cWC[1][1] = B1_row[4]  # None
cWC[1][2] = B1_row[5]
cWC[1][3] = B1_row[6]
cWC[1][4] = B1_row[7]
cWC[1][5] = B1_row[8]
cWC[1][6] = B1_row[9]

# For B2 we follow consistent mapping with context:
# C1: None, C2: None, C3: 100, C4: 50, C5: 50, C6: 100
cWC[2][1] = None
cWC[2][2] = None
cWC[2][3] = B2_row[2]
cWC[2][4] = B2_row[3]
cWC[2][5] = B2_row[4]
cWC[2][6] = B2_row[5]

# For B3: C1: None, C2:150, C3:200, C4:None, C5:50, C6:150
cWC[3][1] = None
cWC[3][2] = B3_row[5]
cWC[3][3] = B3_row[6]
cWC[3][4] = B3_row[7]
cWC[3][5] = B3_row[8]
cWC[3][6] = B3_row[9]

# For B4 no explicit data (unavailable routes)
for k in K:
    cWC[4][k] = None

# B5,B6 -> Ck from C17
B5_row_17 = Table_2_C17[2][1]
B6_row_17 = Table_2_C17[3][1]

cWC[5][1] = B5_row_17[2]
cWC[5][2] = B5_row_17[3]
cWC[5][3] = B5_row_17[4]
cWC[5][4] = B5_row_17[5]
cWC[5][5] = B5_row_17[6]
cWC[5][6] = B5_row_17[7]

cWC[6][1] = B6_row_17[2]
cWC[6][2] = B6_row_17[3]
cWC[6][3] = B6_row_17[4]
cWC[6][4] = B6_row_17[5]
cWC[6][5] = B6_row_17[6]
cWC[6][6] = B6_row_17[7]

# =========================
# 3. Create model
# =========================

model = gp.Model("Warehouse_Selection_Transportation")

# =========================
# 4. Decision variables
# =========================

# Continuous flows
w = model.addVars(I, J, name="w", lb=0.0)   # plant->warehouse
z = model.addVars(I, K, name="z", lb=0.0)   # plant->customer
y = model.addVars(J, K, name="y", lb=0.0)   # warehouse->customer

# Binary warehouse open/built
u = model.addVars(J, vtype=GRB.BINARY, name="u")

# Binary B2 expansion
e2 = model.addVar(vtype=GRB.BINARY, name="e2")

# =========================
# 5. Constraints
# =========================

# 5.1 Route availability: forbid arcs with None cost
for i in I:
    for j in J:
        if cAW[i][j] is None:
            w[i, j].ub = 0.0

for i in I:
    for k in K:
        if cAC[i][k] is None:
            z[i, k].ub = 0.0

for j in J:
    for k in K:
        if cWC[j][k] is None:
            y[j, k].ub = 0.0

# 5.2 User-Origin Preference using PreAssignment structure

# C1 only from A1: so forbid all other sources
# Implementation already: force all warehouse->C1 zero, except none allowed;
# from PreAssignment we ensure plant2 direct is zero.
z[2, 1].ub = 0.0     # C1 not from A2
for j in J:
    y[j, 1].ub = 0.0 # no warehouse to C1 (since C1 must be from A1)

# C2 only from B1
for i in I:
    z[i, 2].ub = 0.0
for j in J:
    if j != 1:
        y[j, 2].ub = 0.0

# C5 only from B2
for i in I:
    z[i, 5].ub = 0.0
for j in J:
    if j != 2:
        y[j, 5].ub = 0.0

# C6 only from B3 or B4
for i in I:
    z[i, 6].ub = 0.0
for j in J:
    if j not in [3, 4]:
        y[j, 6].ub = 0.0

# 5.3 Flow conservation at each warehouse
for j in J:
    model.addConstr(
        gp.quicksum(w[i, j] for i in I) ==
        gp.quicksum(y[j, k] for k in K),
        name=f"FlowCons_{j}"
    )

# 5.4 Plant capacity
for i in I:
    model.addConstr(
        gp.quicksum(w[i, j] for j in J) +
        gp.quicksum(z[i, k] for k in K)
        <= MaxSupply[i],
        name=f"PlantCap_{i}"
    )

# 5.5 Warehouse throughput capacities
T = {}
T[1] = MaxTurnover[1] * u[1]               # 70000 * u1
T[2] = MaxTurnover[2] + Cap_ExpB2 * e2     # 50000 + 20000*e2
T[3] = MaxTurnover[3]                      # 100000 fixed
T[4] = MaxTurnover[4] * u[4]               # 40000 * u4
T[5] = Cap_B5 * u[5]                       # 30000 * u5
T[6] = Cap_B6 * u[6]                       # 25000 * u6

for j in J:
    model.addConstr(
        gp.quicksum(w[i, j] for i in I) +
        gp.quicksum(y[j, k] for k in K)
        <= T[j],
        name=f"WhCap_{j}"
    )

# 5.6 Demand satisfaction for each customer
for k in K:
    model.addConstr(
        gp.quicksum(z[i, k] for i in I) +
        gp.quicksum(y[j, k] for j in J)
        == MinDemand[k],
        name=f"Demand_{k}"
    )

# 5.7 Max number of warehouses
model.addConstr(
    gp.quicksum(u[j] for j in J) <= MaxOpenWarehouses,
    name="MaxWarehouses"
)

# 5.8 Existing warehouses B2,B3 must be open
model.addConstr(u[2] == 1, name="B2_open")
model.addConstr(u[3] == 1, name="B3_open")

# =========================
# 6. Objective function
# =========================

# Transportation cost
transport_cost = gp.LinExpr()
for i in I:
    for j in J:
        if cAW[i][j] is not None:
            transport_cost += cAW[i][j] * w[i, j]
for i in I:
    for k in K:
        if cAC[i][k] is not None:
            transport_cost += cAC[i][k] * z[i, k]
for j in J:
    for k in K:
        if cWC[j][k] is not None:
            transport_cost += cWC[j][k] * y[j, k]

# Investment and savings
investment_cost = Inv_B5 * u[5] + Inv_B6 * u[6] + Inv_ExpB2 * e2
saving_term = -Save_B1 * (1 - u[1]) - Save_B4 * (1 - u[4])

obj_expr = transport_cost + investment_cost + saving_term

model.setObjective(obj_expr, GRB.MINIMIZE)

# =========================
# 7. Solve and print results
# =========================

model.Params.OutputFlag = 0
model.optimize()

FinalAnswer = float('nan')
if model.status == GRB.OPTIMAL:
    FinalAnswer = model.objVal
    print(f"Optimal total cost = {model.objVal:.2f}\n")

    print("Warehouse open decisions (u_j):")
    for j in J:
        print(f"  B{j}: {int(round(u[j].X))}")

    print(f"\nB2 expansion decision (e2): {int(round(e2.X))}\n")

    print("Plant -> Warehouse shipments w[i,j] (t/month):")
    for i in I:
        for j in J:
            if w[i, j].X > 1e-6:
                print(f"  w[{i},{j}] = {w[i,j].X:.2f}")

    print("\nPlant -> Customer shipments z[i,k] (t/month):")
    for i in I:
        for k in K:
            if z[i, k].X > 1e-6:
                print(f"  z[{i},{k}] = {z[i,k].X:.2f}")

    print("\nWarehouse -> Customer shipments y[j,k] (t/month):")
    for j in J:
        for k in K:
            if y[j, k].X > 1e-6:
                print(f"  y[{j},{k}] = {y[j,k].X:.2f}")
else:
    print("Model did not find an optimal solution.")

# Required final answer output
print(f"FinalAnswer=【{FinalAnswer}】")