import gurobipy as gp

# Define all parameter matrices and data inputs
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

# Create model
model = gp.Model("TransportationOptimization")

# Create decision variables
factories = ['P1', 'P2', 'P3', 'P4']
demand_locs = ['D1', 'D2', 'D3', 'D4']

x = {}
for i in factories:
    for j in demand_locs:
        x[i, j] = model.addVar(lb=0, ub=MaxRouteCapacity, name=f"x_{i}_{j}")

# Set up the objective function
obj = gp.quicksum(TransportCost[i][j] * x[i, j] for i in factories for j in demand_locs)
model.setObjective(obj, gp.GRB.MINIMIZE)

# Add all constraints
# 1. Supply capacity constraints
for i in factories:
    model.addConstr(gp.quicksum(x[i, j] for j in demand_locs) <= SupplyCapacity[i], 
                    name=f"Supply_{i}")

# 2. Demand satisfaction constraints
for j in demand_locs:
    model.addConstr(gp.quicksum(x[i, j] for i in factories) >= Demand[j], 
                    name=f"Demand_{j}")

# 3. Route capacity is already enforced by variable bounds (0 <= x <= 2000)

# 4. Minimum P1→D1 agreement
model.addConstr(x['P1', 'D1'] >= MinShip_P1_D1, name="Min_P1_D1")

# 5. Minimum P4→D1 agreement
model.addConstr(x['P4', 'D1'] >= MinShip_P4_D1, name="Min_P4_D1")

# 6. Joint priority on D3 (P2 and P3 must supply at least 20% of D3's demand)
model.addConstr(x['P2', 'D3'] + x['P3', 'D3'] >= MinJointPct_D3 * Demand['D3'], 
                name="Joint_Priority_D3")

# 7. Southern share for P4 (shipments to D2 and D3 ≥ 70% of P4's total shipments)
model.addConstr(x['P4', 'D2'] + x['P4', 'D3'] >= 
                SouthMinPct_P4 * (x['P4', 'D1'] + x['P4', 'D2'] + x['P4', 'D3'] + x['P4', 'D4']),
                name="South_Min_P4")

# 8. Southern limit for P2 (shipments to D2 and D3 ≤ 50% of P2's total shipments)
model.addConstr(x['P2', 'D2'] + x['P2', 'D3'] <= 
                SouthMaxPct_P2 * (x['P2', 'D1'] + x['P2', 'D2'] + x['P2', 'D3'] + x['P2', 'D4']),
                name="South_Max_P2")

# Solve the model
model.optimize()

# Print results
if model.status == gp.GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"Minimum total transportation cost: ${model.ObjVal:,.2f}")
    
    print("\nOptimal shipment quantities:")
    for i in factories:
        for j in demand_locs:
            if x[i, j].X > 0.001:
                print(f"x_{i}_{j} = {x[i, j].X:.2f} tons")
    
    # Calculate and print constraint satisfaction
    print("\nConstraint satisfaction:")
    for i in factories:
        total_ship = sum(x[i, j].X for j in demand_locs)
        print(f"Factory {i} total shipments: {total_ship:.2f} (Capacity: {SupplyCapacity[i]})")
    
    for j in demand_locs:
        total_received = sum(x[i, j].X for i in factories)
        print(f"Demand {j} total received: {total_received:.2f} (Demand: {Demand[j]})")
    
    # Calculate P4 southern share
    p4_total = sum(x['P4', j].X for j in demand_locs)
    p4_south = x['P4', 'D2'].X + x['P4', 'D3'].X
    if p4_total > 0:
        p4_south_pct = (p4_south / p4_total) * 100
        print(f"P4 southern share (D2+D3): {p4_south_pct:.1f}% (Required: ≥{SouthMinPct_P4*100}%)")
    
    # Calculate P2 southern share
    p2_total = sum(x['P2', j].X for j in demand_locs)
    p2_south = x['P2', 'D2'].X + x['P2', 'D3'].X
    if p2_total > 0:
        p2_south_pct = (p2_south / p2_total) * 100
        print(f"P2 southern share (D2+D3): {p2_south_pct:.1f}% (Required: ≤{SouthMaxPct_P2*100}%)")
    
    # Print the final answer
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("No optimal solution found.")
    print(f"FinalAnswer=【Infeasible】")