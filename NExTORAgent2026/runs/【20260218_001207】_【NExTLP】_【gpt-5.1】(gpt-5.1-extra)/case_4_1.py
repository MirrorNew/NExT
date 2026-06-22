import gurobipy as gp
from gurobipy import GRB

# ======================
# Parameter definitions
# ======================

num_factories = 4
num_demand_locations = 4
factories = ['P1', 'P2', 'P3', 'P4']
demand_locations = ['D1', 'D2', 'D3', 'D4']
factory_full_names = [
    'Penang Factory P1',
    'Kuala Lumpur Factory P2',
    'Ipoh Factory P3',
    'Malacca Factory P4'
]
demand_full_names = [
    'Butterworth Transit Terminal D1',
    'Kuantan Manufacturing Park D2',
    'Klang Chemical Park D3',
    'Muar Distribution Center D4'
]
supply = [1000, 1500, 1200, 800]
demand = [900, 1100, 1000, 500]
route_capacity_max = 2000
min_shipment_P1_D1 = 300
min_shipment_P4_D1 = 200
min_share_P2_P3_for_D3 = 0.2
south_share_min_P4 = 0.7
south_share_max_P2 = 0.5
cost = [
    [10, 15, 12, 20],
    [14, 13, 16, 18],
    [12, 17, 11, 15],
    [19, 14, 18, 13]
]

# ======================
# Model
# ======================

model = gp.Model("PVC_Transportation_Optimization")

# ======================
# Decision variables
# x[i,j] = quantity shipped from factory i to demand location j
# indices: i in {0..3} -> P1..P4, j in {0..3} -> D1..D4
# ======================

x = model.addVars(
    num_factories,
    num_demand_locations,
    lb=0.0,
    ub=route_capacity_max,
    vtype=GRB.CONTINUOUS,
    name="x"
)

# ======================
# Objective: minimize total transportation cost
# ======================

model.setObjective(
    gp.quicksum(
        cost[i][j] * x[i, j]
        for i in range(num_factories)
        for j in range(num_demand_locations)
    ),
    GRB.MINIMIZE
)

# ======================
# Constraints
# ======================

# 1. Supply capacity for each factory
for i in range(num_factories):
    model.addConstr(
        gp.quicksum(x[i, j] for j in range(num_demand_locations)) <= supply[i],
        name=f"Supply_capacity_{factories[i]}"
    )

# 2. Demand satisfaction for each demand location
for j in range(num_demand_locations):
    model.addConstr(
        gp.quicksum(x[i, j] for i in range(num_factories)) >= demand[j],
        name=f"Demand_satisfaction_{demand_locations[j]}"
    )

# (Route capacity upper bound is already captured by ub=route_capacity_max in variable definition)

# 3. Contractual minimum shipments
# P1 to D1 >= min_shipment_P1_D1
model.addConstr(
    x[0, 0] >= min_shipment_P1_D1,
    name="P1_D1_minimum_shipment"
)

# P4 to D1 >= min_shipment_P4_D1
model.addConstr(
    x[3, 0] >= min_shipment_P4_D1,
    name="P4_D1_minimum_shipment"
)

# 4. Priority share for D3 from P2 and P3
# D3 is index j = 2, total demand is demand[2] = 1000
min_P2_P3_D3 = min_share_P2_P3_for_D3 * demand[2]
model.addConstr(
    x[1, 2] + x[2, 2] >= min_P2_P3_D3,
    name="D3_priority_share_P2P3"
)

# 5. Southern market share for P4 (D2 and D3)
# P4 is index i = 3, D2 index j=1, D3 index j=2
# x42 + x43 >= 0.7 * (x41 + x42 + x43 + x44)
# Linearized: 0.3*x[3,1] + 0.3*x[3,2] - 0.7*x[3,0] - 0.7*x[3,3] >= 0
model.addConstr(
    0.3 * x[3, 1] + 0.3 * x[3, 2] - 0.7 * x[3, 0] - 0.7 * x[3, 3] >= 0,
    name="P4_southern_share"
)

# 6. Southern share upper bound for P2
# P2 is index i = 1, D2 index j=1, D3 index j=2
# x22 + x23 <= 0.5 * (x21 + x22 + x23 + x24)
# Linearized: 0.5*x[1,1] + 0.5*x[1,2] - 0.5*x[1,0] - 0.5*x[1,3] <= 0
model.addConstr(
    0.5 * x[1, 1] + 0.5 * x[1, 2] - 0.5 * x[1, 0] - 0.5 * x[1, 3] <= 0,
    name="P2_southern_share"
)

# ======================
# Solve
# ======================

model.optimize()

# ======================
# Output results
# ======================

if model.status == GRB.OPTIMAL:
    print("\nOptimal solution found.")
    print(f"Minimum total transportation cost: {model.objVal:.2f}\n")

    print("Shipment plan (x_ij):")
    for i in range(num_factories):
        for j in range(num_demand_locations):
            val = x[i, j].X
            if abs(val) > 1e-6:
                print(
                    f"{factory_full_names[i]} -> {demand_full_names[j]}: "
                    f"{val:.2f}"
                )
else:
    print(f"\nOptimization ended with status {model.status}.")