import gurobipy as gp
from gurobipy import GRB

# Define parameters from the given list
Workshops = ['A', 'B', 'C', 'D', 'E']
Locations = ['Urban area', 'A', 'B']
MaxWorkshopsPerLocation = 3

AnnualCostSavings = {
    'A': {'Move to A': 100, 'Move to B': 100, 'Do not move': 0},
    'B': {'Move to A': 150, 'Move to B': 200, 'Do not move': 0},
    'C': {'Move to A': 100, 'Move to B': 150, 'Do not move': 0},
    'D': {'Move to A': 200, 'Move to B': 150, 'Do not move': 0},
    'E': {'Move to A': 50, 'Move to B': 150, 'Do not move': 0}
}

C_ik = {
    'A': {'B': 0, 'C': 1000, 'D': 1500, 'E': 0},
    'B': {'B': 0, 'C': 1400, 'D': 1200, 'E': 0},
    'C': {'B': 0, 'C': 0, 'D': 0, 'E': 2000},
    'D': {'B': 0, 'C': 0, 'D': 0, 'E': 700}
}

D_jl = {
    'A': {'A': 500, 'B': 1400, 'Urban area': 1300},
    'B': {'A': 1400, 'B': 500, 'Urban area': 900},
    'Urban area': {'A': 1300, 'B': 900, 'Urban area': 1000}
}

# Create a mapping from location names to indices for easier access
location_index = {loc: idx for idx, loc in enumerate(Locations)}
workshop_index = {ws: idx for idx, ws in enumerate(Workshops)}

# Convert savings data to a matrix format
s = {}
for ws in Workshops:
    s[ws] = {}
    s[ws]['Urban area'] = AnnualCostSavings[ws]['Do not move']
    s[ws]['A'] = AnnualCostSavings[ws]['Move to A']
    s[ws]['B'] = AnnualCostSavings[ws]['Move to B']

# Create model
model = gp.Model("WorkshopRelocation")

# Decision variables
x = {}  # x[i][j]
for i in Workshops:
    for j in Locations:
        x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

y = {}  # y[i][k][j][l]
# Only for i < k (ordered pairs)
workshop_pairs = []
for idx_i, i in enumerate(Workshops):
    for idx_k, k in enumerate(Workshops):
        if idx_i < idx_k:
            workshop_pairs.append((i, k))

for i, k in workshop_pairs:
    for j in Locations:
        for l in Locations:
            y[i, k, j, l] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k}_{j}_{l}")

# Set objective
obj_expr = gp.QuadExpr()

# Transportation cost part
for i, k in workshop_pairs:
    C_val = C_ik[i][k] if k in C_ik[i] else 0
    if C_val > 0:  # Only add if there's transportation volume
        for j in Locations:
            for l in Locations:
                D_val = D_jl[j][l]
                obj_expr.add(C_val * D_val * y[i, k, j, l])

# Subtract cost savings part
for i in Workshops:
    for j in Locations:
        obj_expr.add(-s[i][j] * x[i, j])

model.setObjective(obj_expr, GRB.MINIMIZE)

# Constraints

# 1. Each workshop must be assigned to exactly one location
for i in Workshops:
    lhs = gp.LinExpr()
    for j in Locations:
        lhs.add(x[i, j])
    model.addConstr(lhs == 1, name=f"Assign_{i}")

# 2. At most MaxWorkshopsPerLocation workshops per location
for j in Locations:
    lhs = gp.LinExpr()
    for i in Workshops:
        lhs.add(x[i, j])
    model.addConstr(lhs <= MaxWorkshopsPerLocation, name=f"Capacity_{j}")

# 3. Linearization constraints for y[i,k,j,l] = x[i,j] * x[k,l]
for i, k in workshop_pairs:
    for j in Locations:
        for l in Locations:
            # y <= x[i,j]
            model.addConstr(y[i, k, j, l] <= x[i, j], name=f"Lin1_{i}_{k}_{j}_{l}")
            # y <= x[k,l]
            model.addConstr(y[i, k, j, l] <= x[k, l], name=f"Lin2_{i}_{k}_{j}_{l}")
            # y >= x[i,j] + x[k,l] - 1
            model.addConstr(y[i, k, j, l] >= x[i, j] + x[k, l] - 1, name=f"Lin3_{i}_{k}_{j}_{l}")

# Solve the model
model.optimize()

# Print results
print("Optimal solution found!")
print(f"Objective value (total cost): {model.ObjVal:.2f}")

# Calculate transportation cost and savings separately
transport_cost = 0
total_savings = 0

for i, k in workshop_pairs:
    C_val = C_ik[i][k] if k in C_ik[i] else 0
    if C_val > 0:
        for j in Locations:
            for l in Locations:
                if y[i, k, j, l].X > 0.5:
                    D_val = D_jl[j][l]
                    transport_cost += C_val * D_val * y[i, k, j, l].X

for i in Workshops:
    for j in Locations:
        if x[i, j].X > 0.5:
            total_savings += s[i][j]

print(f"Transportation cost increment: {transport_cost:.2f}")
print(f"Total annual cost savings: {total_savings:.2f}")

# Print assignment results
print("\nWorkshop assignments:")
for i in Workshops:
    for j in Locations:
        if x[i, j].X > 0.5:
            print(f"  Workshop {i} -> {j}")

# Count workshops per location
print("\nNumber of workshops per location:")
for j in Locations:
    count = sum(x[i, j].X for i in Workshops)
    print(f"  {j}: {count}")

# Print transportation flows (only for pairs with non-zero volume)
print("\nTransportation flows (only for pairs with C_ik > 0):")
for i, k in workshop_pairs:
    C_val = C_ik[i][k] if k in C_ik[i] else 0
    if C_val > 0:
        for j in Locations:
            for l in Locations:
                if y[i, k, j, l].X > 0.5:
                    D_val = D_jl[j][l]
                    cost = C_val * D_val
                    print(f"  {i}({j}) to {k}({l}): {C_val} tons × {D_val} yuan/ton = {cost:.2f} yuan")

# The question asks for "total cost minus the transportation cost" which is actually the objective value
# But based on the problem statement, we need to output the optimal relocation plan's overall cost
# The objective is Z = transport_increment - savings, which is what we minimized
final_answer = model.ObjVal
print(f"\nFinalAnswer=【{final_answer:.2f}】")