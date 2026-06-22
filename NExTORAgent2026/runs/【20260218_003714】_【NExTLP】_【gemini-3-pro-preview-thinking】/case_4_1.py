import gurobipy as gp
from gurobipy import GRB

# Define the parameters as specified
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

# Southern locations are defined as D2 and D3 in the problem description
Southern_Locations = ['D2', 'D3']

# Factories and Demand Locations lists
Factories = list(SupplyCapacity.keys())
DemandLocations = list(Demand.keys())

# Create the model
model = gp.Model("Chemical_Transportation_Optimization")

# Decision Variables
# x[i,j] = Shipment quantity from factory i to demand location j
x = model.addVars(Factories, DemandLocations, lb=0, ub=MaxRouteCapacity, vtype=GRB.CONTINUOUS, name="x")

# Objective Function
# Minimize total transportation cost
model.setObjective(
    gp.quicksum(TransportCost[i][j] * x[i,j] for i in Factories for j in DemandLocations),
    GRB.MINIMIZE
)

# Constraints

# 1. Supply Capacity Constraints
for i in Factories:
    model.addConstr(gp.quicksum(x[i,j] for j in DemandLocations) <= SupplyCapacity[i], name=f"Supply_{i}")

# 2. Demand Satisfaction Constraints
for j in DemandLocations:
    model.addConstr(gp.quicksum(x[i,j] for i in Factories) >= Demand[j], name=f"Demand_{j}")

# 3. Route Capacity (Already handled by variable upper bound ub=MaxRouteCapacity, but adding explicit for completeness)
for i in Factories:
    for j in DemandLocations:
        model.addConstr(x[i,j] <= MaxRouteCapacity, name=f"RouteCap_{i}_{j}")

# 4. Minimum P1 -> D1
model.addConstr(x['P1', 'D1'] >= MinShip_P1_D1, name="MinShip_P1_D1")

# 5. Minimum P4 -> D1
model.addConstr(x['P4', 'D1'] >= MinShip_P4_D1, name="MinShip_P4_D1")

# 6. Joint Priority on D3 (P2 and P3 must supply at least 20% of D3's demand)
# Note: The problem says "20% of D3's demand", so we use the parameter Demand['D3']
model.addConstr(x['P2', 'D3'] + x['P3', 'D3'] >= MinJointPct_D3 * Demand['D3'], name="JointPriority_D3")

# 7. Southern Share for P4
# Shipments from P4 to southern locations (D2, D3) >= 70% of total shipments from P4
P4_Total_Shipment = gp.quicksum(x['P4', j] for j in DemandLocations)
P4_Southern_Shipment = gp.quicksum(x['P4', j] for j in Southern_Locations)
model.addConstr(P4_Southern_Shipment >= SouthMinPct_P4 * P4_Total_Shipment, name="SouthernShare_P4")

# 8. Southern Limit for P2
# Shipments from P2 to southern locations (D2, D3) <= 50% of total shipments from P2
P2_Total_Shipment = gp.quicksum(x['P2', j] for j in DemandLocations)
P2_Southern_Shipment = gp.quicksum(x['P2', j] for j in Southern_Locations)
model.addConstr(P2_Southern_Shipment <= SouthMaxPct_P2 * P2_Total_Shipment, name="SouthernLimit_P2")

# Solve the model
model.optimize()

# Output results
if model.status == GRB.OPTIMAL:
    print("Optimal Solution Found:")
    for i in Factories:
        for j in DemandLocations:
            if x[i,j].x > 0:
                print(f"Ship from {i} to {j}: {x[i,j].x:.2f}")
    
    print(f"\nFinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found.")
    print(f"FinalAnswer=【No Solution】")