import gurobipy as gp
from gurobipy import GRB

# 1. Create the Gurobi model
model = gp.Model("BinhaiHarborFerryOptimization")

# 2. Define parameter matrices and data inputs
# Nodes: 1, 2, 3, 4, 5, 6, 7
nodes = [1, 2, 3, 4, 5, 6, 7]

# Origin points and mapping to node IDs
# q=1 is residential area 1, q=2 is residential area 4, q=3 is residential area 5
q_values = [1, 2, 3]
q_to_node = {1: 1, 2: 4, 3: 5}

# Arcs and their distances (costs)
# We assume the ferry distance (2, 6) is 0 to calculate reduction in driving mileage.
# The baseline mileage (399250) only accounts for land distances.
costs = {
    (1, 2): 3.5, (2, 1): 3.5,
    (2, 3): 3.0, (3, 2): 3.0,
    (3, 4): 5.0, (4, 3): 5.0,
    (4, 5): 25.0, (5, 4): 25.0,
    (5, 6): 4.0, (6, 5): 4.0,
    (6, 7): 2.5, (7, 6): 2.5,
    (2, 6): 0.0, (6, 2): 0.0
}
arcs = list(costs.keys())

# Demand data from Table C-35 (b_qk: flow from origin node to arrival node k)
# Mapping rows: '1' -> Origin Node 1, '4' -> Origin Node 4, '5' -> Origin Node 5
b_data = {
    1: {2: 900, 3: 750, 4: 40, 5: 10, 6: 600, 7: 550},
    4: {1: 100, 2: 2000, 3: 1100, 5: 150, 6: 1400, 7: 1250},
    5: {1: 110, 2: 4000, 3: 2200, 4: 200, 6: 3300, 7: 2440}
}

# Calculate net supply for each origin at each node
s = {}
for q in q_values:
    origin_node = q_to_node[q]
    for i in nodes:
        if i == origin_node:
            s[q, i] = sum(b_data[origin_node].values())
        elif i in b_data[origin_node]:
            s[q, i] = -b_data[origin_node][i]
        else:
            s[q, i] = 0

# 3. Create decision variables
# x[q, i, j] is the flow of people/vehicles from origin q on arc (i, j)
x = model.addVars(q_values, arcs, lb=0, vtype=GRB.CONTINUOUS, name="x")

# 4. Set up the objective function
# Minimize total travel mileage after the ferry is opened
total_mileage_with_ferry = gp.quicksum(costs[i, j] * x[q, i, j] for q in q_values for (i, j) in arcs)
model.setObjective(total_mileage_with_ferry, GRB.MINIMIZE)

# 5. Add constraints
# Flow conservation for each origin q and each node i
for q in q_values:
    for i in nodes:
        model.addConstr(
            gp.quicksum(x[q, i, j] for j in nodes if (i, j) in arcs) -
            gp.quicksum(x[q, j, i] for j in nodes if (j, i) in arcs) == s[q, i],
            name=f"flow_bal_q{q}_node{i}"
        )

# Ferry capacity constraints
# The ferry can transport 2000 cars in each direction (2->6 and 6->2)
ferry_cap = 2000
model.addConstr(gp.quicksum(x[q, 2, 6] for q in q_values) <= ferry_cap, name="ferry_cap_2_6")
model.addConstr(gp.quicksum(x[q, 6, 2] for q in q_values) <= ferry_cap, name="ferry_cap_6_2")

# 6. Solve the model
model.optimize()

# 7. Print results and calculate reduction
if model.status == GRB.OPTIMAL:
    min_mileage_after = model.objVal
    baseline_mileage = 399250
    reduction = baseline_mileage - min_mileage_after
    
    print(f"Total Mileage Before Ferry: {baseline_mileage}")
    print(f"Total Mileage After Ferry: {min_mileage_after}")
    print(f"Total Mileage Reduced: {reduction}")
    print(f"FinalAnswer=【{int(reduction)}】")
else:
    print("Optimization was not successful.")