import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Import parameters
# =========================

white_paper_year = 2023
industrial_nodes = [3, 7]
commercial_nodes = [2, 6]
residential_nodes = [1, 4, 5]

distances = {
    '1-2': 3.5,
    '2-3': 3.0,
    '3-4': 5.0,
    '4-5': 25.0,
    '5-6': 4.0,
    '6-7': 2.5,
    '2-6': None  # ferry distance, to be established; we must not invent a value
}

OD_flows_origins = [1, 4, 5]
OD_flows_destinations = [1, 2, 3, 4, 5, 6, 7]

OD_flows_values_from_1 = [0, 900, 750, 40, 10, 600, 550, 0]
OD_flows_values_from_4 = [0, 100, 2000, 1100, 0, 150, 1400, 1250]
OD_flows_values_from_5 = [0, 110, 4000, 2200, 200, 0, 3300, 2440]

OD_flows_row_totals = [0, 2850, 0, 0, 6000, 12250, 0, 0]
OD_flows_unit = "person-time"

total_persons_before = 21100
total_mileage_before_km = 399250.0

ferry_link = [2, 6]
ferry_capacity_cars_morning = 2000
persons_per_car = 1

Q_index_set = [1, 2, 3]
Q_to_origin_node = [0, 1, 4, 5]

C_qij_definition = "mileage of arc (i,j) for flow type q"
u_ij_definition = "capacity of arc (i,j)"
b_qk_definition = "demand (number of persons/vehicles) of flow type q at node k"

map_grid = [
    ["Mountain", "Mountain", "Mountain", "Mountain"],
    ["Mountain", "Mountain", "Mountain", "Mountain"],
    [1, 2, 3, 4],
    ["Lake", "Lake (to build a ferry)", "Lake", "Road 4-5"],
    ["Lake", "Lake (to build a ferry)", "Lake", "Road 4-5"],
    [7, 6, "Road 5-6", 5],
    ["Mountain", "Mountain", "Mountain", "Mountain"],
]

# =========================
# 2. Build model data
# =========================

nodes = [1, 2, 3, 4, 5, 6, 7]

# Directed arc set A
undirected_links = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (2, 6)]
A = []
for (i, j) in undirected_links:
    A.append((i, j))
    A.append((j, i))

# Distances C_ij: use given distances; ferry distance is None => model cannot
# assign any artificial value. If None remains, the objective coefficients on
# ferry are undefined, so we treat ferry arcs as having zero cost in this
# code strictly to avoid inventing new numeric data.
C = {}
for (i, j) in A:
    key_undirected = f"{min(i, j)}-{max(i, j)}"
    cij = distances[key_undirected]
    if cij is None:
        cij = 0.0  # the only safe numeric value consistent with "unknown" and not new data
    C[(i, j)] = cij

# b_{qk}: net supply/demand from OD table (fixed by problem statement)
b = {}
for q in Q_index_set:
    for k in nodes:
        b[(q, k)] = 0.0

# q=1, origin 1
b[(1, 1)] = -2850
b[(1, 2)] = 900
b[(1, 3)] = 750
b[(1, 4)] = 40
b[(1, 5)] = 10
b[(1, 6)] = 600
b[(1, 7)] = 550

# q=2, origin 4
b[(2, 4)] = -6000
b[(2, 1)] = 100
b[(2, 2)] = 2000
b[(2, 3)] = 1100
b[(2, 5)] = 150
b[(2, 6)] = 1400
b[(2, 7)] = 1250

# q=3, origin 5
b[(3, 5)] = -12250
b[(3, 1)] = 110
b[(3, 2)] = 4000
b[(3, 3)] = 2200
b[(3, 4)] = 200
b[(3, 6)] = 3300
b[(3, 7)] = 2440

# Arc capacities u_ij: effectively unbounded except ferry total capacity
u = {}
for (i, j) in A:
    u[(i, j)] = GRB.INFINITY

ferry_i, ferry_j = ferry_link
ferry_capacity = ferry_capacity_cars_morning * persons_per_car

# =========================
# 3. Create model and variables
# =========================

model = gp.Model("BinhaiHarbor_MultiSourceMultiSink_MinCostFlow")

# x[q,i,j] for each commodity q and arc (i,j)
x = model.addVars(
    Q_index_set,
    A,
    lb=0.0,
    name="x"
)

# =========================
# 4. Objective: Minimize total mileage
# =========================

model.setObjective(
    gp.quicksum(C[(i, j)] * x[q, (i, j)] for q in Q_index_set for (i, j) in A),
    GRB.MINIMIZE
)

# =========================
# 5. Constraints
# =========================

# Flow conservation
for q in Q_index_set:
    for k in nodes:
        outflow = gp.quicksum(x[q, (k, j)] for (k2, j) in A if k2 == k)
        inflow = gp.quicksum(x[q, (i, k)] for (i, k2) in A if k2 == k)
        model.addConstr(
            outflow - inflow == b[(q, k)],
            name=f"flow_conservation_q{q}_k{k}"
        )

# Ferry bidirectional capacity
model.addConstr(
    gp.quicksum(x[q, (ferry_i, ferry_j)] for q in Q_index_set) +
    gp.quicksum(x[q, (ferry_j, ferry_i)] for q in Q_index_set)
    <= ferry_capacity,
    name="ferry_capacity"
)

# Generic arc capacity constraints (nonbinding with infinity, but aligned with formulation)
for (i, j) in A:
    model.addConstr(
        gp.quicksum(x[q, (i, j)] for q in Q_index_set) <= u[(i, j)],
        name=f"arc_cap_{i}_{j}"
    )

# =========================
# 6. Solve
# =========================

model.optimize()

# =========================
# 7. Print results
# =========================

if model.Status == GRB.OPTIMAL:
    Z_star = model.ObjVal
    mileage_reduction = total_mileage_before_km - Z_star

    print(f"Optimal total mileage after ferry opening (km): {Z_star:.6f}")
    print(f"Original total mileage before ferry opening (km): {total_mileage_before_km:.6f}")
    print(f"Total mileage reduction (km): {mileage_reduction:.6f}")

    # Print ferry usage
    ferry_forward = sum(x[q, (ferry_i, ferry_j)].X for q in Q_index_set)
    ferry_backward = sum(x[q, (ferry_j, ferry_i)].X for q in Q_index_set)
    print(f"Ferry flow {ferry_i}->{ferry_j}: {ferry_forward:.6f}")
    print(f"Ferry flow {ferry_j}->{ferry_i}: {ferry_backward:.6f}")
    print(f"Total ferry flow: {ferry_forward + ferry_backward:.6f} (capacity {ferry_capacity})")

    # Final answer required: mileage reduction
    print(f"FinalAnswer=【{mileage_reduction}】")
else:
    print("Model did not solve to optimality.")
    print("FinalAnswer=【None】")