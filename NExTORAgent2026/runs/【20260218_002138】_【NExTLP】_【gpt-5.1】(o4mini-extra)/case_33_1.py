import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters and data
# =========================

# From Parameters List (note: residents[0] is dummy to keep 1-based indexing)
max_nursing_homes = 4
residents = [0.0, 5.2, 4.4, 7.1, 9.0, 6.1, 5.7, 10.0, 12.2, 7.6,
             20.3, 30.4, 30.9, 12.0, 9.3, 15.5, 25.6, 11.0, 5.3, 7.9, 9.9]

num_regions = 20
num_sites = 10

regions = range(1, num_regions + 1)
sites = range(1, num_sites + 1)

# Coverage matrix a_{ij}: 1 if site j can serve region i, 0 otherwise
# Initialize with zeros
a = {(i, j): 0 for i in regions for j in sites}

# Fill in coverage according to the given adjacency (nonzero a_{ij})
# Site 1: regions 2, 3
a[(2, 1)] = 1
a[(3, 1)] = 1

# Site 2: regions 1, 2, 6, 7
a[(1, 2)] = 1
a[(2, 2)] = 1
a[(6, 2)] = 1
a[(7, 2)] = 1

# Site 3: regions 3, 4, 5, 8
a[(3, 3)] = 1
a[(4, 3)] = 1
a[(5, 3)] = 1
a[(8, 3)] = 1

# Site 4: regions 7, 8, 10, 11, 12, 13
a[(7, 4)] = 1
a[(8, 4)] = 1
a[(10, 4)] = 1
a[(11, 4)] = 1
a[(12, 4)] = 1
a[(13, 4)] = 1

# Site 5: regions 11, 12, 13, 16, 17
a[(11, 5)] = 1
a[(12, 5)] = 1
a[(13, 5)] = 1
a[(16, 5)] = 1
a[(17, 5)] = 1

# Site 6: regions 10, 12, 15, 16
a[(10, 6)] = 1
a[(12, 6)] = 1
a[(15, 6)] = 1
a[(16, 6)] = 1

# Site 7: regions 13, 17
a[(13, 7)] = 1
a[(17, 7)] = 1

# Site 8: regions 9, 14, 18
a[(9, 8)] = 1
a[(14, 8)] = 1
a[(18, 8)] = 1

# Site 9: regions 14, 15, 18, 19
a[(14, 9)] = 1
a[(15, 9)] = 1
a[(18, 9)] = 1
a[(19, 9)] = 1

# Site 10: regions 17, 19, 20
a[(17, 10)] = 1
a[(19, 10)] = 1
a[(20, 10)] = 1


# =========================
# 2. Create model
# =========================

model = gp.Model("ElderlyCare_NursingHome_Location")

# =========================
# 3. Decision variables
# =========================

# x_j = 1 if nursing home at site j is built
x = model.addVars(sites, vtype=GRB.BINARY, name="x")

# y_i = 1 if region i is covered by at least one built nursing home
y = model.addVars(regions, vtype=GRB.BINARY, name="y")

# =========================
# 4. Objective function
# =========================

# Minimize total population not covered: sum_i p_i * (1 - y_i)
model.setObjective(
    gp.quicksum(residents[i] * (1 - y[i]) for i in regions),
    GRB.MINIMIZE
)

# =========================
# 5. Constraints
# =========================

# Budget constraint: at most max_nursing_homes nursing homes
model.addConstr(
    gp.quicksum(x[j] for j in sites) <= max_nursing_homes,
    name="Budget"
)

# Coverage definition: a region can be covered only if some serving site is built
for i in regions:
    model.addConstr(
        y[i] <= gp.quicksum(a[(i, j)] * x[j] for j in sites),
        name=f"CoverDef_{i}"
    )

# =========================
# 6. Optimize
# =========================

model.optimize()

# =========================
# 7. Print results
# =========================

if model.Status == GRB.OPTIMAL:
    print("\nOptimal objective value (total population not covered, in 10,000 people):",
          model.ObjVal)

    # Selected nursing home sites
    opened_sites = [j for j in sites if x[j].X > 0.5]
    print("Selected nursing home sites (indices 1-10):", opened_sites)

    # Covered regions
    covered_regions = [i for i in regions if y[i].X > 0.5]
    print("Covered regions (indices 1-20):", covered_regions)

    # Total population not covered (should equal objective)
    total_uncovered_population = model.ObjVal

else:
    # If no optimal solution, set output to None
    total_uncovered_population = None
    print("No optimal solution found. Model status:", model.Status)

# =========================
# 8. Final Answer output
# =========================
# The question asks: "Where should they be built to minimize the total population not covered?"
# The required FinalAnswer is a single value, so we return the minimum total population not covered.
print(f"FinalAnswer=【{total_uncovered_population}】")