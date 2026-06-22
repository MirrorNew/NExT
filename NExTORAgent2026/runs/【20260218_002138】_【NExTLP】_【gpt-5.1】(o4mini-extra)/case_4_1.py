import gurobipy as gp
from gurobipy import GRB

# ==========================
# 1. Define parameters (from Parameters List)
# ==========================

SupplyCapacity = {'P1': 1000, 'P2': 1500, 'P3': 1200, 'P4': 800}

Demand = {'D1': 900, 'D2': 1100, 'D3': 1000, 'D4': 500}

TransportCost = {
    'P1': {'D1': 10, 'D2': 15, 'D3': 12, 'D4': 20},
    'P2': {'D1': 14, 'D2': 13, 'D3': 16, 'D4': 18},
    'P3': {'D1': 12, 'D2': 17, 'D3': 11, 'D4': 15},
    'P4': {'D1': 19, 'D2': 14, 'D3': 18, 'D4': 13}
}

MaxRouteCapacity = 2000
MinShip_P1_D1 = 300
MinShip_P4_D1 = 200
MinJointPct_D3 = 0.2
SouthMinPct_P4 = 0.7
SouthMaxPct_P2 = 0.5

plants = list(SupplyCapacity.keys())      # ['P1','P2','P3','P4']
demands = list(Demand.keys())             # ['D1','D2','D3','D4']

# ==========================
# 2. Create model
# ==========================

model = gp.Model("PVC_Transport_Optimization")

# ==========================
# 3. Decision variables
# ==========================

# x[i,j] : shipment quantity from plant i to demand j
x = model.addVars(
    plants,
    demands,
    vtype=GRB.CONTINUOUS,
    lb=0.0,
    ub=MaxRouteCapacity,
    name="x"
)

# ==========================
# 4. Objective function: Minimize total transport cost
# ==========================

model.setObjective(
    gp.quicksum(TransportCost[i][j] * x[i, j] for i in plants for j in demands),
    sense=GRB.MINIMIZE
)

# ==========================
# 5. Constraints
# ==========================

# 5.1 Supply capacity: sum_j x[i,j] <= SupplyCapacity[i]
for i in plants:
    model.addConstr(
        gp.quicksum(x[i, j] for j in demands) <= SupplyCapacity[i],
        name=f"Supply_{i}"
    )

# 5.2 Demand satisfaction: sum_i x[i,j] >= Demand[j]
for j in demands:
    model.addConstr(
        gp.quicksum(x[i, j] for i in plants) >= Demand[j],
        name=f"Demand_{j}"
    )

# 5.3 Route capacity is already handled by variable upper bound MaxRouteCapacity

# 5.4 Minimum P1 -> D1
model.addConstr(
    x['P1', 'D1'] >= MinShip_P1_D1,
    name="MinShip_P1_D1"
)

# 5.5 Minimum P4 -> D1
model.addConstr(
    x['P4', 'D1'] >= MinShip_P4_D1,
    name="MinShip_P4_D1"
)

# 5.6 Joint priority on D3: x[P2,D3] + x[P3,D3] >= MinJointPct_D3 * Demand['D3']
model.addConstr(
    x['P2', 'D3'] + x['P3', 'D3'] >= MinJointPct_D3 * Demand['D3'],
    name="JointPriority_D3"
)

# 5.7 Southern share for P4:
# x[P4,D2] + x[P4,D3] >= SouthMinPct_P4 * (x[P4,D1] + x[P4,D2] + x[P4,D3] + x[P4,D4])
model.addConstr(
    x['P4', 'D2'] + x['P4', 'D3'] >= SouthMinPct_P4 * (
        x['P4', 'D1'] + x['P4', 'D2'] + x['P4', 'D3'] + x['P4', 'D4']
    ),
    name="SouthMinShare_P4"
)

# 5.8 Southern limit for P2:
# x[P2,D2] + x[P2,D3] <= SouthMaxPct_P2 * (x[P2,D1] + x[P2,D2] + x[P2,D3] + x[P2,D4])
model.addConstr(
    x['P2', 'D2'] + x['P2', 'D3'] <= SouthMaxPct_P2 * (
        x['P2', 'D1'] + x['P2', 'D2'] + x['P2', 'D3'] + x['P2', 'D4']
    ),
    name="SouthMaxShare_P2"
)

# No indicator constraints are needed for this model.

# ==========================
# 6. Solve the model
# ==========================

model.optimize()

# ==========================
# 7. Print results
# ==========================

if model.status == GRB.OPTIMAL:
    print("Optimal solution found.")
    print(f"Minimum total transportation cost: {model.objVal:.2f}")
    print("\nShipment plan (x[i,j]):")
    for i in plants:
        for j in demands:
            val = x[i, j].X
            if abs(val) > 1e-6:
                print(f"  From {i} to {j}: {val:.2f}")
else:
    print(f"Optimization ended with status {model.status}")

# According to the problem statement, the requested answer is:
# "Only the lowest total transportation cost needs to be given."
# So FinalAnswer is the optimal objective value (total transport cost).
if model.status == GRB.OPTIMAL:
    FinalAnswer = model.objVal
else:
    FinalAnswer = None

print(f"FinalAnswer=【{FinalAnswer}】")