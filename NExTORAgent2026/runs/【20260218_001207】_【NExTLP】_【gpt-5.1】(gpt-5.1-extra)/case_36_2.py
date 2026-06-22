import gurobipy as gp
from gurobipy import GRB

# ===============================
# 1. Parameters and data
# ===============================

# Use the provided Parameters List exactly
num_residential_communities = 12
num_schools = 3
radius_m = 500.0
num_product_groups = 4
product_group_names = [
    'Tobacco, alcohol and food',
    'Vegetables and fruits',
    'Rice, noodles and non-staple food',
    'Daily necessities'
]
product_group_short_names = [
    'tobacco and alcohol group',
    'vegetable and fruit group',
    'grain and non-staple food group',
    'general merchandise group'
]
block_labels = ['I', 'II', 'III', 'IV']

Table_1_Customer_Flow_Thousand_People_Per_Day = [
    [None, 5, 2, 7],
    [5, None, 3, 8],
    [2, 3, None, 3],
    [7, 8, 3, None]
]

Table_2_Average_Distance_Between_Blocks_m = [
    [None, 20.0, 37.5, 42.5],
    [20.0, None, 32.5, 25.0],
    [37.5, 32.5, None, 30.0],
    [42.5, 25.0, 30.0, None]
]

# Index sets
P = range(num_product_groups)          # product groups 0..3
B = range(len(block_labels))           # blocks 0..3

# Build numeric matrices F (flow) and D (distance), replace None with 0
F = [[0.0 for _ in P] for __ in P]
D = [[0.0 for _ in B] for __ in B]

for p in P:
    for q in P:
        val = Table_1_Customer_Flow_Thousand_People_Per_Day[p][q]
        F[p][q] = 0.0 if val is None else float(val)

for i in B:
    for j in B:
        val = Table_2_Average_Distance_Between_Blocks_m[i][j]
        D[i][j] = 0.0 if val is None else float(val)

# ===============================
# 2. Create model
# ===============================

model = gp.Model("Supermarket_Block_Assignment")

# Nonconvex quadratic terms x[p,i] * x[q,j] in the objective
model.Params.NonConvex = 2

# ===============================
# 3. Decision variables
# ===============================

# x[p,i] = 1 if product group p is assigned to block i; 0 otherwise
x = model.addVars(
    P, B,
    vtype=GRB.BINARY,
    name="x"
)

# ===============================
# 4. Auxiliary substitution variables
#    Free continuous variables, linked to product x[p,i] * x[q,j]
# ===============================

# y[p,q,i,j] is an auxiliary variable to substitute the product x[p,i] * x[q,j]
# Domain: (-inf, +inf)
y = model.addVars(
    P, P, B, B,
    lb=-GRB.INFINITY,
    ub=GRB.INFINITY,
    vtype=GRB.CONTINUOUS,
    name="y"
)

# Link y and x: y[p,q,i,j] = x[p,i] * x[q,j]
for p in P:
    for q in P:
        for i in B:
            for j in B:
                model.addConstr(
                    y[p, q, i, j] == x[p, i] * x[q, j],
                    name=f"link_y_{p}_{q}_{i}_{j}"
                )

# ===============================
# 5. Objective function
# ===============================

# Mathematical expression (from context, round-trip distance):
#   min Z = 2 * ∑_{p,q,i,j} F_{pq} · D_{ij} · x_{p,i} · x_{q,j}
# Implement via auxiliary y variables:
objective = gp.quicksum(
    2.0 * F[p][q] * D[i][j] * y[p, q, i, j]
    for p in P for q in P for i in B for j in B
)

model.setObjective(objective, GRB.MINIMIZE)

# ===============================
# 6. Constraints
# ===============================

# (1) Product-to-block assignment: each product to exactly one block
for p in P:
    model.addConstr(
        gp.quicksum(x[p, i] for i in B) == 1,
        name=f"assign_product_{p}"
    )

# (2) Block capacity: each block holds exactly one product
for i in B:
    model.addConstr(
        gp.quicksum(x[p, i] for p in P) == 1,
        name=f"assign_block_{i}"
    )

# ===============================
# 7. Solve the model and print results
# ===============================

model.optimize()

if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
    print("\nOptimal assignment of product groups to blocks:")
    assignment = {}
    for p in P:
        for i in B:
            if x[p, i].X > 0.5:
                assignment[p] = i
                print(
                    f"  Product group '{product_group_short_names[p]}'"
                    f" -> Block {block_labels[i]}"
                )

    optimal_value = model.ObjVal
    print(
        f"\nMinimum total daily round-trip walking distance: "
        f"{optimal_value:.4f}"
    )

    # FinalAnswer is the requested minimized total round-trip distance
    print(f"FinalAnswer=【{optimal_value:.4f}】")
else:
    print("No optimal solution found.")
    print("FinalAnswer=【NaN】")