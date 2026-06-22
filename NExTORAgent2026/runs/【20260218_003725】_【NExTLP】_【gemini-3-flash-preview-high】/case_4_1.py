import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs
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

# Factory and Location identifiers
factories = list(SupplyCapacity.keys())
locations = list(Demand.keys())

# 2. Create the model
model = gp.Model("TransportationCostOptimization")

# 3. Create decision variables
# x_i_j: Shipment quantity from factory i to demand location j
x = model.addVars(factories, locations, lb=0, ub=MaxRouteCapacity, name="x")

# 4. Set up the objective function
# Minimize total transportation cost
total_cost = gp.quicksum(TransportCost[i][j] * x[i, j] for i in factories for j in locations)
model.setObjective(total_cost, GRB.MINIMIZE)

# 5. Add all constraints
# Supply Capacity: sum_j x_i_j <= S_i for each factory i
for i in factories:
    model.addConstr(gp.quicksum(x[i, j] for j in locations) <= SupplyCapacity[i], name=f"Supply_{i}")

# Demand Satisfaction: sum_i x_i_j >= D_j for each demand location j
for j in locations:
    model.addConstr(gp.quicksum(x[i, j] for i in factories) >= Demand[j], name=f"Demand_{j}")

# Contractual requirement: P1 to D1 at least MinShip_P1_D1 units
model.addConstr(x['P1', 'D1'] >= MinShip_P1_D1, name="Contract_P1_D1")

# Contractual requirement: P4 to D1 at least MinShip_P4_D1 units
model.addConstr(x['P4', 'D1'] >= MinShip_P4_D1, name="Contract_P4_D1")

# Joint Priority on D3: at least 20% of D3's demand must be jointly undertaken by P2 and P3
model.addConstr(x['P2', 'D3'] + x['P3', 'D3'] >= MinJointPct_D3 * Demand['D3'], name="JointPriority_D3")

# Southern Share for P4: volume to D2 and D3 must be >= 70% of P4's total shipments
model.addConstr(x['P4', 'D2'] + x['P4', 'D3'] >= SouthMinPct_P4 * gp.quicksum(x['P4', j] for j in locations), name="SouthernShare_P4")

# Southern Limit for P2: volume to D2 and D3 must be <= 50% of P2's total shipments
model.addConstr(x['P2', 'D2'] + x['P2', 'D3'] <= SouthMaxPct_P2 * gp.quicksum(x['P2', j] for j in locations), name="SouthernLimit_P2")

# 6. Solve the model
model.optimize()

# 7. Print results and output final answer
if model.status == GRB.OPTIMAL:
    print(f"Total Transportation Cost: {model.ObjVal}")
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimal solution was not found.")