import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters (from Parameters List)
# =========================
NumRegions = 12
NumSites = 8
MaxStationsToBuild = 6
CoverageTimeLimit = 5.0  # not explicitly used in this MILP, but retained
WeightMinTwoCoverage = 6
WeightMaxTwoCoverage = 8
WeightMinThreeCoverage = 9
ExtraCoverageRegions = [5, 6]
I = [1, 2, 3, 4, 5, 6, 7, 8]  # Sites
J = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Regions

# Index 0 is dummy to keep 1-based indexing consistent with the statement
CoverableAreas = [
    [],
    [1, 2, 5, 6, 7, 8, 11],  # site 1
    [2, 3, 6, 7, 5],         # site 2
    [1, 4, 5, 6, 7, 8],      # site 3
    [3, 4, 5, 6, 7],         # site 4
    [6, 7, 10, 11],          # site 5
    [4, 5, 8, 11, 12],       # site 6
    [2, 9, 11, 12],          # site 7
    [5, 10, 12]              # site 8
]
ConstructionCost = [0, 90, 70, 85, 65, 60, 80, 60, 50]
ImportanceWeight = [0, 4, 8, 5, 10, 7, 9, 6, 6, 3, 5, 8, 4]
Table_1_SiteCoverageAndCost = [
    [1, [1, 2, 5, 6, 7, 8, 11], 90],
    [2, [2, 3, 6, 7, 5], 70],
    [3, [1, 4, 5, 6, 7, 8], 85],
    [4, [3, 4, 5, 6, 7], 65],
    [5, [6, 7, 10, 11], 60],
    [6, [4, 5, 8, 11, 12], 80],
    [7, [2, 9, 11, 12], 60],
    [8, [5, 10, 12], 50]
]
Table_2_RegionImportance = [
    [1, 4],
    [2, 8],
    [3, 5],
    [4, 10],
    [5, 7],
    [6, 9],
    [7, 6],
    [8, 6],
    [9, 3],
    [10, 5],
    [11, 8],
    [12, 4]
]

# =========================
# 2. Preprocessing: coverage matrix by region
# =========================
# For each region j, determine which sites i can cover it
sites_covering_region = {j: [] for j in J}
for i in I:
    for j in CoverableAreas[i]:
        sites_covering_region[j].append(i)

# Determine required coverage level per region based on weight and extra coverage
required_coverage = {}
for j in J:
    w = ImportanceWeight[j]
    if w >= WeightMinThreeCoverage:
        base_cov = 3
    elif WeightMinTwoCoverage <= w <= WeightMaxTwoCoverage:
        base_cov = 2
    else:
        base_cov = 1
    # Extra coverage for library regions (5 and 6)
    if j in ExtraCoverageRegions:
        base_cov += 1
    required_coverage[j] = base_cov

# =========================
# 3. Create model
# =========================
model = gp.Model("XunAn_FireStation_RobustCoverage")

# =========================
# 4. Decision variables
# =========================
# x[i] = 1 if a station is built at site i
x = model.addVars(I, vtype=GRB.BINARY, name="x")

# =========================
# 5. Objective: minimize total construction cost
# =========================
model.setObjective(
    gp.quicksum(ConstructionCost[i] * x[i] for i in I),
    GRB.MINIMIZE
)

# =========================
# 6. Constraints
# =========================

# 6.1 Build limit: at most MaxStationsToBuild stations
model.addConstr(
    gp.quicksum(x[i] for i in I) <= MaxStationsToBuild,
    name="Build_limit"
)

# 6.2 Coverage constraints for each region
for j in J:
    model.addConstr(
        gp.quicksum(x[i] for i in sites_covering_region[j]) >= required_coverage[j],
        name=f"Region_{j}_coverage"
    )

# NOTE: No indicator-variable logic is present in this model,
# so addGenConstrIndicator is not required.

# =========================
# 7. Solve model
# =========================
model.optimize()

# =========================
# 8. Print results
# =========================
if model.Status == GRB.OPTIMAL:
    print("Optimal solution found.")
    print(f"Minimum total construction cost: {model.ObjVal}")
    for i in I:
        print(f"Build station at site {i}: {int(x[i].X)}")
else:
    print(f"Optimization ended with status {model.Status}")

# =========================
# 9. Final answer output
# The question's answer is the minimum total construction cost
# =========================
if model.Status == GRB.OPTIMAL:
    final_answer = model.ObjVal
else:
    final_answer = None

print(f"FinalAnswer=【{final_answer}】")