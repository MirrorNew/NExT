import gurobipy as gp
from gurobipy import GRB

# Create the Gurobi model
model = gp.Model("HarborTransportation")

# --- 1. Define Parameters and Data Inputs ---

# Nodes in the graph
nodes = [1, 2, 3, 4, 5, 6, 7]

# Origins (Residential Areas)
origins = [1, 4, 5]

# Distance/Cost Data for Road Network (Bidirectional)
# Units: km or arbitrary distance units given in the problem
# Mapping: (Node A, Node B): Distance
road_distances = {
    (1, 2): 3.5,
    (2, 3): 3.0,
    (3, 4): 5.0,
    (4, 5): 25.0,
    (5, 6): 4.0,
    (6, 7): 2.5
}

# Demand Data (Person-time / Vehicles, assuming 1 person per car)
# Structure: Origin -> {Destination: Demand}
demands = {
    1: {2: 900, 3: 750, 4: 40, 5: 10, 6: 600, 7: 550},
    4: {1: 100, 2: 2000, 3: 1100, 5: 150, 6: 1400, 7: 1250},
    5: {1: 110, 2: 4000, 3: 2200, 4: 200, 6: 3300, 7: 2440}
}

# Constants
ferry_capacity = 2000  # Vehicles per direction
original_mileage = 399250  # Total mileage before optimization

# Build complete list of arcs and costs
arcs = []
costs = {}

# Add existing roads (both directions)
for (u, v), d in road_distances.items():
    # Forward
    arcs.append((u, v))
    costs[(u, v)] = d
    # Backward
    arcs.append((v, u))
    costs[(v, u)] = d

# Add Ferry Arcs (2 <-> 6)
# Cost is set to 0 to represent 0 driving mileage
ferry_arcs = [(2, 6), (6, 2)]
for u, v in ferry_arcs:
    arcs.append((u, v))
    costs[(u, v)] = 0.0

# Convert arcs to Gurobi tuplelist for efficient indexing
arcs = gp.tuplelist(arcs)

# --- 2. Create Decision Variables ---

# x[q, i, j]: Flow of vehicles from origin q on arc (i, j)
x = model.addVars(origins, arcs, vtype=GRB.CONTINUOUS, name="x", lb=0.0)

# R: Total mileage reduction
R = model.addVar(vtype=GRB.CONTINUOUS, name="R", lb=0.0)

# --- 3. Set up the Objective Function ---

# Maximize the reduction in mileage
model.setObjective(R, GRB.MAXIMIZE)

# --- 4. Add Constraints ---

# (1) Mileage Reduction Definition
# R + New_Total_Mileage = Original_Mileage
current_mileage = gp.quicksum(costs[i,j] * x[q,i,j] for q in origins for i,j in arcs)
model.addConstr(R + current_mileage == original_mileage, name="MileageDef")

# (2) Ferry Capacity Constraints
# "2000 cars in both directions" implies 2000 each way (standard link capacity interpretation)
# Also consistent with context: sum(x...2,6) <= 2000
model.addConstr(gp.quicksum(x[q, 2, 6] for q in origins) <= ferry_capacity, name="FerryCap_2_6")
model.addConstr(gp.quicksum(x[q, 6, 2] for q in origins) <= ferry_capacity, name="FerryCap_6_2")

# (3) Flow Conservation Constraints
# For each origin q, flow balance at every node k
for q in origins:
    q_demands = demands[q]
    total_supply = sum(q_demands.values())  # Total flow originating from q
    
    for k in nodes:
        flow_in = gp.quicksum(x[q, i, k] for i, j in arcs if j == k)
        flow_out = gp.quicksum(x[q, k, j] for i, j in arcs if i == k)
        
        if k == q:
            # Source Node: Net Outflow = Total Supply
            model.addConstr(flow_out - flow_in == total_supply, name=f"FlowBal_Source_{q}_{k}")
        elif k in q_demands:
            # Destination Node: Net Inflow = Demand (or Net Out = -Demand)
            d_qk = q_demands[k]
            model.addConstr(flow_out - flow_in == -d_qk, name=f"FlowBal_Dest_{q}_{k}")
        else:
            # Transshipment Node: Net Flow = 0
            model.addConstr(flow_out - flow_in == 0, name=f"FlowBal_Trans_{q}_{k}")

# --- 5. Solve the Model and Print Results ---

model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"Optimal Solution Found.")
    print(f"Total Reduced Mileage: {R.X}")
    print(f"New Total Mileage: {original_mileage - R.X}")
    print(f"FinalAnswer=【{R.X}】")
else:
    print("No optimal solution found.")