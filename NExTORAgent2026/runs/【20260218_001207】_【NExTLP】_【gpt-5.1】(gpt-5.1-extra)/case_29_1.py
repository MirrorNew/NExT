import gurobipy as gp
from gurobipy import GRB

# ==========================
# 1. Parameters and data
# ==========================

# Use ONLY the given Parameters List values

year_of_establishment = 1998

max_workshops_per_location = 3

workshops = ['A', 'B', 'C', 'D', 'E']

workshop_descriptions = {
    'A': 'precision processing, old equipment, high relocation and transformation costs',
    'B': 'provides semi-finished products for workshop A',
    'C': 'assembly, highly dependent on transportation efficiency',
    'D': 'electroplating, old equipment, high relocation and transformation costs',
    'E': 'warehousing, highly dependent on transportation efficiency'
}

locations = ['Urban area', 'Satellite Town A', 'Satellite Town B']

annual_cost_savings_from_relocation = {
    'A': {'Satellite Town A': 100, 'Satellite Town B': 100, 'Urban area': 0},
    'B': {'Satellite Town A': 150, 'Satellite Town B': 200, 'Urban area': 0},
    'C': {'Satellite Town A': 100, 'Satellite Town B': 150, 'Urban area': 0},
    'D': {'Satellite Town A': 200, 'Satellite Town B': 150, 'Urban area': 0},
    'E': {'Satellite Town A': 50,  'Satellite Town B': 150, 'Urban area': 0}
}

Table_1_annual_cost_savings = [
    ['Workshop', 'Move to A', 'Move to B', 'Do not move'],
    ['A', 100, 100, 0],
    ['B', 150, 200, 0],
    ['C', 100, 150, 0],
    ['D', 200, 150, 0],
    ['E', 50, 150, 0]
]

# Cik matrix: 5x5, order of workshops = ['A','B','C','D','E']
Cik_transport_volume_matrix = [
    [0, 0, 1000, 1500, 0],
    [0, 0, 1400, 1200, 0],
    [0, 0, 0,    0,   2000],
    [0, 0, 0,    0,   700],
    [0, 0, 0,    0,   0]
]

Table_2_Cik_values = [
    ['', 'B', 'C', 'D', 'E'],
    ['A', 0, 1000, 1500, 0],
    ['B', 0, 1400, 1200, 0],
    ['C', 0, 0, 0, 2000],
    ['D', 0, 0, 0, 700]
]

# Djl matrix: 3x3, order of locations = ['A','B','Urban area'] (as in original table)
Djl_unit_transport_cost_matrix = [
    [500, 1400, 1300],
    [1400, 500, 900],
    [1300, 900, 1000]
]

Table_3_Djl_values = [
    ['', 'A', 'B', 'Urban area'],
    ['A', 500, 1400, 1300],
    ['B', 1400, 500, 900],
    ['Urban area', 1300, 900, 1000]
]

Cik_definition = {
    'meaning': 'annual transportation volume between workshops i and k',
    'unit': 'tons/year',
    'condition': 'i < k, Cik = 0 means annual transportation volume between workshops i and k is 0'
}

Djl_definition = {
    'meaning': 'unit transportation cost between locations j and l',
    'unit': 'yuan/ton',
    'symmetry_assumption': 'price from City B to City A equals price from City A to City B'
}

objective_description = {
    'goal': 'minimize overall cost',
    'components': ['annual cost savings (land, rent, sewage, etc.)',
                   'increment in transportation cost after relocation']
}

# --------------------------------------------------
# Build convenient index mappings for Cik and Djl
# --------------------------------------------------

# Map workshop names to indices in Cik matrix
ws_index = {w: i for i, w in enumerate(workshops)}

# Extract non-zero Cik pairs with i<k
pairs = []
Cik = {}
for i, wi in enumerate(workshops):
    for k, wk in enumerate(workshops):
        if i < k:
            val = Cik_transport_volume_matrix[i][k]
            if val != 0:
                pairs.append((wi, wk))
                Cik[(wi, wk)] = val

# Map our model locations to the indices used in Djl_unit_transport_cost_matrix
# Original table order: ['A','B','Urban area']
loc_to_Didx = {
    'Satellite Town A': 0,  # 'A'
    'Satellite Town B': 1,  # 'B'
    'Urban area': 2         # 'Urban area'
}

# Build Dlm in our location naming
Dlm = {}
for l1 in locations:
    for l2 in locations:
        i1 = loc_to_Didx[l1]
        i2 = loc_to_Didx[l2]
        Dlm[(l1, l2)] = Djl_unit_transport_cost_matrix[i1][i2]

# ==========================
# 2. Create model
# ==========================

model = gp.Model("Workshop_Relocation_NLP_Linearized")

# ==========================
# 3. Decision variables
# ==========================

# x[i,l] = 1 if workshop i is located at location l
x = model.addVars(
    workshops,
    locations,
    vtype=GRB.BINARY,
    name="x"
)

# z[i,k,l,m] = 1 if workshop i is at location l AND workshop k is at location m
# For all (i,k) with Cik > 0 and all (l,m) in locations
z = model.addVars(
    [(i, k, l, m) for (i, k) in pairs for l in locations for m in locations],
    vtype=GRB.BINARY,
    name="z"
)

# ==========================
# 4. Auxiliary substitution / indicator vars if needed
# (Here, z already linearizes x_i_l * x_k_m; no extra continuous aux vars needed)
# ==========================

# No nonlinear functions like powers or logs; no NonConvex flag required.

# ==========================
# 5. Objective function
# ==========================

# Maximize total cost savings minus transportation cost increment

# Savings term
saving_expr = gp.quicksum(
    annual_cost_savings_from_relocation[i][l] * x[i, l]
    for i in workshops
    for l in locations
)

# Transportation cost increment term
transport_expr = gp.quicksum(
    Cik[(i, k)] * Dlm[(l, m)] * z[i, k, l, m]
    for (i, k) in pairs
    for l in locations
    for m in locations
)

model.setObjective(saving_expr - transport_expr, GRB.MAXIMIZE)

# ==========================
# 6. Constraints
# ==========================

# 6.1 Assignment constraint: each workshop in exactly one location
for i in workshops:
    model.addConstr(
        gp.quicksum(x[i, l] for l in locations) == 1,
        name=f"assign_{i}"
    )

# 6.2 Capacity constraints: at most 3 workshops per location
for l in locations:
    model.addConstr(
        gp.quicksum(x[i, l] for i in workshops) <= max_workshops_per_location,
        name=f"capacity_{l}".replace(" ", "_")
    )

# 6.3 Linearization constraints for z = x_i_l * x_k_m
# Using standard MILP linearization
for (i, k) in pairs:
    for l in locations:
        for m in locations:
            model.addConstr(z[i, k, l, m] <= x[i, l],
                            name=f"z_le_xi_{i}_{k}_{l}_{m}".replace(" ", "_"))
            model.addConstr(z[i, k, l, m] <= x[k, m],
                            name=f"z_le_xk_{i}_{k}_{l}_{m}".replace(" ", "_"))
            model.addConstr(z[i, k, l, m] >= x[i, l] + x[k, m] - 1,
                            name=f"z_ge_sum_minus1_{i}_{k}_{l}_{m}".replace(" ", "_"))

# ==========================
# 7. Solve model and print results
# ==========================

model.Params.OutputFlag = 0  # Turn off solver output for cleanliness; remove/comment to see log
model.optimize()

if model.SolCount == 0:
    print("No feasible solution found.")
    FinalAnswer = None
else:
    # Print location decisions
    print("Optimal workshop locations:")
    for i in workshops:
        for l in locations:
            if x[i, l].X > 0.5:
                print(f"  Workshop {i} -> {l}")
    # Compute components of the objective for reporting
    total_savings = saving_expr.getValue()
    total_transport = transport_expr.getValue()
    net_benefit = model.ObjVal  # same as total_savings - total_transport

    print(f"\nTotal annual cost savings (land/rent/sewage etc.): {total_savings}")
    print(f"Total annual transportation cost increment: {total_transport}")
    print(f"Net annual benefit (savings - transportation cost): {net_benefit}")

    # The question asks: "give the total cost minus the transportation cost."
    # That is exactly the net benefit (objective value).
    FinalAnswer = net_benefit

# Required final output format
print(f"FinalAnswer=【{FinalAnswer}】")