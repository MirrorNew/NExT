import gurobipy as gp
from gurobipy import GRB

# Define parameters from the Parameters List
total_mileage = 399250
ferry_capacity = 2000
car_occupancy = 1

# Node sets
V = [1, 2, 3, 4, 5, 6, 7]
Q = [1, 2, 3]  # Origins: 1, 4, 5

# Map origin index to actual node
origin_nodes = {1: 1, 2: 4, 3: 5}

# Define arcs (undirected edges, but we need both directions for flow)
edges_undirected = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (2, 6)]
E = []
for (i, j) in edges_undirected:
    E.append((i, j))
    E.append((j, i))

# Distances (costs) - symmetric
C = {}
C[(1, 2)] = C[(2, 1)] = 3.5
C[(2, 3)] = C[(3, 2)] = 3.0
C[(3, 4)] = C[(4, 3)] = 5.0
C[(4, 5)] = C[(5, 4)] = 25.0
C[(5, 6)] = C[(6, 5)] = 4.0
C[(6, 7)] = C[(7, 6)] = 2.5
# For ferry arc (2,6) and (6,2), we need to determine the distance
# Since it's a ferry, we assume it's a direct connection with some distance
# The problem doesn't specify the ferry distance, but we can infer it should be
# less than the land route distance to be attractive
# Let's calculate the shortest land route distance between 2 and 6:
# 2-3-4-5-6: 3 + 5 + 25 + 4 = 37
# Or 2-1? No direct connection to 6 from 1
# The ferry distance should be less than 37. We'll assume a reasonable value.
# From the map, 2 and 6 are separated by lakes, so ferry might be much shorter.
# Let's assume ferry distance is 10 (this is an assumption since not specified).
ferry_distance = 10.0
C[(2, 6)] = C[(6, 2)] = ferry_distance

# Demand parameters b_{qk} from Table C-35
b = {}
# Origin q=1 (residential area 1)
b[(1, 1)] = 0
b[(1, 2)] = 900
b[(1, 3)] = 750
b[(1, 4)] = 40
b[(1, 5)] = 10
b[(1, 6)] = 600
b[(1, 7)] = 550

# Origin q=2 (residential area 4)
b[(2, 1)] = 100
b[(2, 2)] = 2000
b[(2, 3)] = 1100
b[(2, 4)] = 0
b[(2, 5)] = 150
b[(2, 6)] = 1400
b[(2, 7)] = 1250

# Origin q=3 (residential area 5)
b[(3, 1)] = 110
b[(3, 2)] = 4000
b[(3, 3)] = 2200
b[(3, 4)] = 200
b[(3, 5)] = 0
b[(3, 6)] = 3300
b[(3, 7)] = 2440

# Calculate total supply for each origin
supply = {}
for q in Q:
    origin = origin_nodes[q]
    total = 0
    for k in V:
        total += b[(q, k)]
    supply[q] = total

# Create model
model = gp.Model("Ferry_Optimization")

# Decision variables: x_{qij} - flow on arc (i,j) for origin q
x = {}
for q in Q:
    for (i, j) in E:
        x[q, i, j] = model.addVar(lb=0.0, name=f"x_{q}_{i}_{j}")

# R: total mileage reduction
R = model.addVar(lb=0.0, name="R")

# Set objective: maximize R
model.setObjective(R, GRB.MAXIMIZE)

# Add constraints

# 1. Flow conservation for each origin q and each node i
for q in Q:
    origin = origin_nodes[q]
    for i in V:
        outflow = gp.quicksum(x[q, i, j] for (i2, j) in E if i2 == i)
        inflow = gp.quicksum(x[q, j, i] for (j, i2) in E if i2 == i)
        
        if i == origin:
            # Supply node: outflow - inflow = total supply
            model.addConstr(outflow - inflow == supply[q], 
                          name=f"flow_cons_supply_q{q}_node{i}")
        else:
            # Demand node: outflow - inflow = -b_{q,i}
            model.addConstr(outflow - inflow == -b[(q, i)], 
                          name=f"flow_cons_demand_q{q}_node{i}")

# 2. Ferry capacity constraints (arcs (2,6) and (6,2))
model.addConstr(gp.quicksum(x[q, 2, 6] for q in Q) <= ferry_capacity,
                name="ferry_capacity_2_6")
model.addConstr(gp.quicksum(x[q, 6, 2] for q in Q) <= ferry_capacity,
                name="ferry_capacity_6_2")

# 3. Mileage reduction definition: R + sum_{q,i,j} C_{ij} * x_{qij} = total_mileage
total_flow_cost = gp.quicksum(C[(i, j)] * x[q, i, j] 
                             for q in Q for (i, j) in E)
model.addConstr(R + total_flow_cost == total_mileage,
                name="mileage_reduction_def")

# Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print(f"Optimal solution found")
    print(f"Total mileage reduction R = {R.X:.2f}")
    print(f"Total flow cost = {total_flow_cost.getValue():.2f}")
    
    # Check ferry usage
    ferry_flow_26 = sum(x[q, 2, 6].X for q in Q)
    ferry_flow_62 = sum(x[q, 6, 2].X for q in Q)
    print(f"Ferry flow (2->6): {ferry_flow_26:.2f}")
    print(f"Ferry flow (6->2): {ferry_flow_62:.2f}")
    
    # Calculate total flow for verification
    total_flow = 0
    for q in Q:
        for (i, j) in E:
            total_flow += x[q, i, j].X
    print(f"Total flow in network: {total_flow:.2f}")
    
    # Output the answer to the question
    print(f"FinalAnswer=【{R.X:.2f}】")
else:
    print(f"No optimal solution found. Status: {model.status}")
    print(f"FinalAnswer=【0】")