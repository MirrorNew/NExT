import gurobipy as gp
from gurobipy import GRB

# Define the parameters as specified in the Parameters List
workshops = ['A', 'B', 'C', 'D', 'E']
locations = ['Urban area', 'A', 'B']
max_workshops_per_location = 3

annual_cost_savings_data = {
    'A': {'Move to A': 100, 'Move to B': 100, 'Do not move': 0},
    'B': {'Move to A': 150, 'Move to B': 200, 'Do not move': 0},
    'C': {'Move to A': 100, 'Move to B': 150, 'Do not move': 0},
    'D': {'Move to A': 200, 'Move to B': 150, 'Do not move': 0},
    'E': {'Move to A': 50, 'Move to B': 150, 'Do not move': 0}
}

# Transportation volume C_ik (i to k)
c_ik_data = {
    'A': {'B': 0, 'C': 1000, 'D': 1500, 'E': 0},
    'B': {'B': 0, 'C': 1400, 'D': 1200, 'E': 0},
    'C': {'B': 0, 'C': 0, 'D': 0, 'E': 2000},
    'D': {'B': 0, 'C': 0, 'D': 0, 'E': 700}
}

# Unit transportation cost D_jl
d_jl_data = {
    'A': {'A': 500, 'B': 1400, 'Urban area': 1300},
    'B': {'A': 1400, 'B': 500, 'Urban area': 900},
    'Urban area': {'A': 1300, 'B': 900, 'Urban area': 1000}
}

# Create the model
model = gp.Model("WorkshopRelocation")

# Pre-process data into easier lookup structures

# 1. Savings S[i][j]
# Map location names to the keys in annual_cost_savings_data
loc_key_map = {
    'Urban area': 'Do not move',
    'A': 'Move to A',
    'B': 'Move to B'
}
savings = {}
for i in workshops:
    for j in locations:
        savings[i, j] = annual_cost_savings_data[i][loc_key_map[j]]

# 2. Transport Volume C[i][k] (extract non-zero pairs)
# The data provided is for i < k roughly, or directed flow. 
# We assume the flow is from source (outer key) to dest (inner key).
transport_flows = []
for i in c_ik_data:
    for k in c_ik_data[i]:
        vol = c_ik_data[i][k]
        if vol > 0:
            transport_flows.append((i, k, vol))

# 3. Unit Cost D[j][l]
# Direct lookup is sufficient, but let's ensure consistency
unit_costs = {}
for j in locations:
    for l in locations:
        unit_costs[j, l] = d_jl_data[j][l]

# --- Decision Variables ---

# x[i, j] = 1 if workshop i is assigned to location j
x = model.addVars(workshops, locations, vtype=GRB.BINARY, name="x")

# y[i, k, j, l] = 1 if workshop i is at j AND workshop k is at l
# Only created for pairs (i, k) that have transport volume
y = {}
for (i, k, vol) in transport_flows:
    for j in locations:
        for l in locations:
            y[i, k, j, l] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k}_{j}_{l}")

# --- Objective Function ---

# Minimize Total Cost = (Transportation Increment) - (Cost Savings)
# Transportation Increment = sum(C_ik * D_jl * y_ikjl)
transport_cost = 0
for (i, k, vol) in transport_flows:
    for j in locations:
        for l in locations:
            transport_cost += vol * unit_costs[j, l] * y[i, k, j, l]

# Cost Savings = sum(S_ij * x_ij)
cost_savings = gp.quicksum(savings[i, j] * x[i, j] for i in workshops for j in locations)

model.setObjective(transport_cost - cost_savings, GRB.MINIMIZE)

# --- Constraints ---

# 1. Assignment Constraint: Each workshop must be assigned to exactly one location
model.addConstrs(
    (x.sum(i, '*') == 1 for i in workshops),
    name="AssignmentConstraint"
)

# 2. Location Capacity Constraint: No more than 3 workshops per location
model.addConstrs(
    (x.sum('*', j) <= max_workshops_per_location for j in locations),
    name="LocationCapacityConstraint"
)

# 3. Linearization Constraints for y[i,k,j,l] = x[i,j] * x[k,l]
# Constraints:
# y <= x[i,j]
# y <= x[k,l]
# y >= x[i,j] + x[k,l] - 1
for (i, k, vol) in transport_flows:
    for j in locations:
        for l in locations:
            y_var = y[i, k, j, l]
            model.addConstr(y_var <= x[i, j], name=f"LinearizationUpper1_{i}_{k}_{j}_{l}")
            model.addConstr(y_var <= x[k, l], name=f"LinearizationUpper2_{i}_{k}_{j}_{l}")
            model.addConstr(y_var >= x[i, j] + x[k, l] - 1, name=f"LinearizationLower_{i}_{k}_{j}_{l}")

# --- Solve and Output ---
model.optimize()

if model.status == GRB.OPTIMAL:
    print("\nOptimal Solution Found:")
    for i in workshops:
        for j in locations:
            if x[i, j].X > 0.5:
                print(f"Workshop {i} assigned to {j}")
    
    # Calculate components for verification
    trans_val = 0
    for (i, k, vol) in transport_flows:
        for j in locations:
            for l in locations:
                if y[i, k, j, l].X > 0.5:
                    trans_val += vol * unit_costs[j, l]
    
    sav_val = 0
    for i in workshops:
        for j in locations:
            if x[i, j].X > 0.5:
                sav_val += savings[i, j]
                
    print(f"Total Transportation Cost: {trans_val}")
    print(f"Total Annual Savings: {sav_val}")
    print(f"Net Cost (Objective): {model.objVal}")
    
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found.")