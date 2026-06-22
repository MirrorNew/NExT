import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs
workshops = ['A', 'B', 'C', 'D', 'E']
locations = ['Urban area', 'A', 'B']
max_workshops_per_location = 3

# Annual cost savings: s[i][j]
# Table C-18 Mapping:
# 'Urban area' corresponds to 'Do not move'
# 'A' corresponds to 'Move to A'
# 'B' corresponds to 'Move to B'
savings_data = {
    'A': {'Move to A': 100, 'Move to B': 100, 'Do not move': 0},
    'B': {'Move to A': 150, 'Move to B': 200, 'Do not move': 0},
    'C': {'Move to A': 100, 'Move to B': 150, 'Do not move': 0},
    'D': {'Move to A': 200, 'Move to B': 150, 'Do not move': 0},
    'E': {'Move to A': 50, 'Move to B': 150, 'Do not move': 0}
}
s = {
    w: {
        'A': savings_data[w]['Move to A'],
        'B': savings_data[w]['Move to B'],
        'Urban area': savings_data[w]['Do not move']
    } for w in workshops
}

# Annual transportation volume: C[i][k] (tons/year)
# Table C-19: Only i < k are considered based on the table's definition
C_ik_data = {
    ('A', 'B'): 0, ('A', 'C'): 1000, ('A', 'D'): 1500, ('A', 'E'): 0,
    ('B', 'C'): 1400, ('B', 'D'): 1200, ('B', 'E'): 0,
    ('C', 'D'): 0, ('C', 'E'): 2000,
    ('D', 'E'): 700
}

# Unit transportation cost: D[j][l] (yuan/ton)
# Table C-20: Data provided in a symmetric fashion but mapped specifically
D_jl_data = {
    'A': {'A': 500, 'B': 1400, 'Urban area': 1300},
    'B': {'A': 1400, 'B': 500, 'Urban area': 900},
    'Urban area': {'A': 1300, 'B': 900, 'Urban area': 1000}
}

# 2. Create the model and variables
model = gp.Model("WorkshopRelocationOptimization")

# x[i][j] = 1 if workshop i is assigned to location j
x = model.addVars(workshops, locations, vtype=GRB.BINARY, name="x")

# y[i][k][j][l] = 1 if workshop i is at location j and workshop k is at location l
y = {}
for (i, k) in C_ik_data:
    for j in locations:
        for l in locations:
            y[i, k, j, l] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k}_{j}_{l}")

# 3. Set up the objective function
# Overall cost = (New Transportation Cost - Original Transportation Cost) - Relocation Savings
# Original Transportation Cost assumes all workshops start in the 'Urban area'
original_transport_cost = sum(vol * D_jl_data['Urban area']['Urban area'] for vol in C_ik_data.values())

new_transport_cost = gp.quicksum(
    C_ik_data[i, k] * D_jl_data[j][l] * y[i, k, j, l]
    for (i, k) in C_ik_data
    for j in locations
    for l in locations
)

total_relocation_savings = gp.quicksum(
    s[i][j] * x[i, j]
    for i in workshops
    for j in locations
)

# Total overall cost to minimize (increment - savings)
model.setObjective(new_transport_cost - original_transport_cost - total_relocation_savings, GRB.MINIMIZE)

# 4. Add constraints
# Each workshop must be assigned to exactly one location
for i in workshops:
    model.addConstr(gp.quicksum(x[i, j] for j in locations) == 1, name=f"Assign_{i}")

# No more than 3 workshops per location
for j in locations:
    model.addConstr(gp.quicksum(x[i, j] for i in workshops) <= max_workshops_per_location, name=f"Capacity_{j}")

# Quadratic term linearization using Indicator Constraints: y[i,k,j,l] = x[i,j] * x[k,l]
for (i, k) in C_ik_data:
    for j in locations:
        for l in locations:
            # Indicator logic for logical AND: y=1 iff x1=1 AND x2=1
            model.addGenConstrIndicator(y[i, k, j, l], 1, x[i, j] + x[k, l] == 2)
            model.addGenConstrIndicator(y[i, k, j, l], 0, x[i, j] + x[k, l] <= 1)

# 5. Solve the model and print results
model.optimize()

if model.status == GRB.OPTIMAL:
    # Print the objective value as requested
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("Optimization was not successful.")