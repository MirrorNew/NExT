import gurobipy as gp
from gurobipy import GRB

# =======================
# 1. Define parameters
# =======================

# Sets
Workshops = ['A', 'B', 'C', 'D', 'E']
Locations = ['Urban area', 'A', 'B']

MaxWorkshopsPerLocation = 3

AnnualCostSavings = {
    'A': {'Move to A': 100, 'Move to B': 100, 'Do not move': 0},
    'B': {'Move to A': 150, 'Move to B': 200, 'Do not move': 0},
    'C': {'Move to A': 100, 'Move to B': 150, 'Do not move': 0},
    'D': {'Move to A': 200, 'Move to B': 150, 'Do not move': 0},
    'E': {'Move to A': 50,  'Move to B': 150, 'Do not move': 0}
}

C_ik = {
    'A': {'B': 0,    'C': 1000, 'D': 1500, 'E': 0},
    'B': {'B': 0,    'C': 1400, 'D': 1200, 'E': 0},
    'C': {'B': 0,    'C': 0,    'D': 0,    'E': 2000},
    'D': {'B': 0,    'C': 0,    'D': 0,    'E': 700}
}

D_jl = {
    'A':          {'A': 500,  'B': 1400, 'Urban area': 1300},
    'B':          {'A': 1400, 'B': 500,  'Urban area': 900},
    'Urban area': {'A': 1300, 'B': 900,  'Urban area': 1000}
}

# Build a convenience matrix for savings s[i][j] as in the model
s = {i: {} for i in Workshops}
for i in Workshops:
    s[i]['Urban area'] = AnnualCostSavings[i]['Do not move']
    s[i]['A'] = AnnualCostSavings[i]['Move to A']
    s[i]['B'] = AnnualCostSavings[i]['Move to B']

# =======================
# 2. Create model
# =======================

model = gp.Model("WorkshopRelocation")

# =======================
# 3. Decision variables
# =======================

# x[i,j] = 1 if workshop i is assigned to location j
x = model.addVars(
    Workshops,
    Locations,
    vtype=GRB.BINARY,
    name="x"
)

# y[i,k,j,l] = 1 if workshops (i,k) are assigned to (j,l), with i<k
pairs = [(i, k) for idx_i, i in enumerate(Workshops)
         for k in Workshops[idx_i+1:]]  # i<k
y = model.addVars(
    [(i, k, j, l) for (i, k) in pairs for j in Locations for l in Locations],
    vtype=GRB.BINARY,
    name="y"
)

# =======================
# 4. Objective function
# =======================

# Transportation cost increment term:
transport_cost = gp.quicksum(
    C_ik[i][k] * D_jl[j][l] * y[i, k, j, l]
    for (i, k) in pairs
    for j in Locations
    for l in Locations
    if C_ik.get(i, {}).get(k, 0) != 0  # skip zero-volume pairs to save time
)

# Savings term:
savings = gp.quicksum(
    s[i][j] * x[i, j]
    for i in Workshops
    for j in Locations
)

# Minimize (transport cost increment - savings)
model.setObjective(transport_cost - savings, GRB.MINIMIZE)

# =======================
# 5. Constraints
# =======================

# 5.1 Assignment: each workshop is assigned to exactly one location
for i in Workshops:
    model.addConstr(
        gp.quicksum(x[i, j] for j in Locations) == 1,
        name=f"Assign_{i}"
    )

# 5.2 Capacity: no more than MaxWorkshopsPerLocation per location
for j in Locations:
    model.addConstr(
        gp.quicksum(x[i, j] for i in Workshops) <= MaxWorkshopsPerLocation,
        name=f"Capacity_{j}"
    )

# 5.3 Linearization constraints for y = x_i_j * x_k_l
for (i, k) in pairs:
    for j in Locations:
        for l in Locations:
            # Upper bounds
            model.addConstr(
                y[i, k, j, l] <= x[i, j],
                name=f"LinUB1_{i}_{k}_{j}_{l}"
            )
            model.addConstr(
                y[i, k, j, l] <= x[k, l],
                name=f"LinUB2_{i}_{k}_{j}_{l}"
            )
            # Lower bound
            model.addConstr(
                y[i, k, j, l] >= x[i, j] + x[k, l] - 1,
                name=f"LinLB_{i}_{k}_{j}_{l}"
            )

# =======================
# 6. Optimize
# =======================

model.optimize()

# =======================
# 7. Print results
# =======================

if model.status == GRB.OPTIMAL:
    print("Optimal objective (total cost = transport increment - savings):", model.objVal)
    print("\nWorkshop locations:")
    for i in Workshops:
        for j in Locations:
            if x[i, j].X > 0.5:
                print(f"  Workshop {i} -> {j}")

    # Compute total transport cost and total savings separately for reporting
    total_transport_cost = sum(
        C_ik[i][k] * D_jl[j][l] * y[i, k, j, l].X
        for (i, k) in pairs
        for j in Locations
        for l in Locations
        if C_ik.get(i, {}).get(k, 0) != 0
    )

    total_savings = sum(
        s[i][j] * x[i, j].X
        for i in Workshops
        for j in Locations
    )

    print(f"\nTotal transport cost increment: {total_transport_cost}")
    print(f"Total annual cost savings: {total_savings}")
    print(f"Objective value (increment - savings): {total_transport_cost - total_savings}")

    # The question asks for "total cost minus the transportation cost".
    # That equals: (transport increment - savings) - (transport increment) = -savings.
    total_cost_minus_transport_cost = -total_savings
    print(f"\nTotal cost minus the transportation cost: {total_cost_minus_transport_cost}")

    # FinalAnswer must output this quantity
    print(f"FinalAnswer=【{total_cost_minus_transport_cost}】")
else:
    print("No optimal solution found.")
    # In case of no optimal solution, set FinalAnswer to None
    print("FinalAnswer=【None】")