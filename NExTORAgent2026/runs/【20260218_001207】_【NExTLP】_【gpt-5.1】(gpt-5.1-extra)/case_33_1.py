import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Import parameters from the given Parameters List
# =========================

number_of_regions = 20
population_unit = 10000
number_of_candidate_nursing_homes = 10
max_number_of_nursing_homes_to_build = 4

Table_1_C34 = [None, 5.2, 4.4, 7.1, 9.0, 6.1, 5.7, 10.0, 12.2, 7.6,
               20.3, 30.4, 30.9, 12.0, 9.3, 15.5, 25.6, 11.0, 5.3, 7.9, 9.9]

Table_2_Coverage = [
    None,
    [2],            # region 1
    [1, 2],         # region 2
    [1, 3],         # region 3
    [3],            # region 4
    [3],            # region 5
    [2],            # region 6
    [2, 4],         # region 7
    [3, 4],         # region 8
    [8],            # region 9
    [4, 6],         # region 10
    [4, 5],         # region 11
    [4, 5, 6],      # region 12
    [4, 5, 7],      # region 13
    [8, 9],         # region 14
    [6, 9],         # region 15
    [5, 6],         # region 16
    [5, 7, 10],     # region 17
    [8, 9],         # region 18
    [9, 10],        # region 19
    [10]            # region 20
]

Table_3_Adjacency = [
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]
]

# Index sets (1-based)
regions = range(1, number_of_regions + 1)
facilities = range(1, number_of_candidate_nursing_homes + 1)

# =========================
# 2. Create Gurobi model
# =========================
model = gp.Model("Jinxiu_Elderly_Care_Phase1")

# =========================
# 3. Decision variables
# =========================
# x_j: 1 if nursing home j is built
x = model.addVars(facilities, vtype=GRB.BINARY, name="x")

# y_i: 1 if region i is covered by at least one built nursing home
y = model.addVars(regions, vtype=GRB.BINARY, name="y")

# =========================
# 4. Objective function
# Minimize total uncovered population: sum_i p_i * (1 - y_i)
# =========================
model.setObjective(
    gp.quicksum(Table_1_C34[i] * (1 - y[i]) for i in regions),
    GRB.MINIMIZE
)

# =========================
# 5. Constraints
# =========================

# (1) Facility budget: at most max_number_of_nursing_homes_to_build facilities
model.addConstr(
    gp.quicksum(x[j] for j in facilities) <= max_number_of_nursing_homes_to_build,
    name="Facility_budget"
)

# (2) Coverage linking: y_i <= sum_{j in N(i)} x_j  for all regions
for i in regions:
    Ni = Table_2_Coverage[i]  # list of facilities that can serve region i
    model.addConstr(
        y[i] <= gp.quicksum(x[j] for j in Ni),
        name=f"Coverage_linking_{i}"
    )

# Binary domains are already enforced by vtype=GRB.BINARY

# =========================
# 6. Solve the model
# =========================
model.Params.OutputFlag = 0  # suppress solver log (optional)
model.optimize()

# =========================
# 7. Extract and print results
# =========================
if model.status == GRB.OPTIMAL:
    # Objective value: minimum uncovered population in 10,000 people
    min_uncovered_population_10k = model.objVal

    built_facilities = [j for j in facilities if x[j].X > 0.5]
    covered_regions = [i for i in regions if y[i].X > 0.5]
    uncovered_regions = [i for i in regions if y[i].X <= 0.5]

    print("Optimal solution found.")
    print(f"Minimum uncovered population (10,000 people): {min_uncovered_population_10k:.4f}")
    print(f"Minimum uncovered population (people): {min_uncovered_population_10k * population_unit:.0f}")
    print("Nursing homes to build (indices 1-10):", built_facilities)
    print("Covered regions:", covered_regions)
    print("Uncovered regions:", uncovered_regions)

    # The question asks: "Where should they be built? ... to minimize the total population not covered?"
    # We return the indices of the selected nursing homes as the final answer.
    the_question_answer = built_facilities
else:
    print("No optimal solution found.")
    the_question_answer = None

# Required final output line
print(f"FinalAnswer=【{the_question_answer}】")