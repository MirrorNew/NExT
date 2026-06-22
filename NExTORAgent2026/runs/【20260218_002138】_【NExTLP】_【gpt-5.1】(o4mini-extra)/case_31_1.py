import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters (from Parameters List)
# =========================
Year_established = 1995
Total_screened_candidates = 180
Category_3_count = 30
Total_recruits = 170
Qualified_applicants = 180
Categories_count = 6
p1 = 10000
p2 = 100
p3 = 1
p2_threshold = 0.8
p3_threshold = 0.8
i_max = 6
j_max = 3
k_max = 2
major_index_mapping = [None, 'Production', 'Marketing', 'Finance']
city_index_mapping = [None, 'A', 'B']
goals_count = 7
majors = ['Production', 'Marketing', 'Finance']
cities = ['A', 'B']
Table_1_C_24 = [
    ['A', 'Production', 20],
    ['A', 'Marketing', 30],
    ['A', 'Finance', 40],
    ['B', 'Production', 25],
    ['B', 'Marketing', 20],
    ['B', 'Finance', 35]
]
Table_2_C_25 = [
    [1, 30, ['Production', 'Marketing'], 'Production', 'A'],
    [2, 30, ['Marketing', 'Finance'], 'Marketing', 'A'],
    [3, 30, ['Production', 'Finance'], 'Production', 'B'],
    [4, 30, ['Production', 'Finance'], 'Finance', 'B'],
    [5, 30, ['Marketing', 'Finance'], 'Finance', 'A'],
    [6, 30, ['Finance'], 'Finance', 'B']
]

# =========================
# 2. Derived data structures
# =========================

# Map major and city names to indices used in x_{ijk}
major_to_j = {name: idx for idx, name in enumerate(major_index_mapping) if idx is not None}
city_to_k = {name: idx for idx, name in enumerate(city_index_mapping) if idx is not None}

# Quotas per (city, major) from Table_1_C_24
quota = {}  # (j,k) -> required number
for city_name, major_name, num in Table_1_C_24:
    j = major_to_j[major_name]
    k = city_to_k[city_name]
    quota[(j, k)] = num

# Category info from Table_2_C_25
category_supply = {}        # i -> number of people (max 30)
category_suitable = {}      # i -> set of suitable major indices S_i
category_desired_major = {} # i -> j_desired(i)
category_desired_city = {}  # i -> k_desired(i)

for row in Table_2_C_25:
    i_cat, cnt, suitable_list, desired_major_name, desired_city_name = row
    category_supply[i_cat] = cnt
    category_suitable[i_cat] = {major_to_j[m] for m in suitable_list}
    category_desired_major[i_cat] = major_to_j[desired_major_name]
    category_desired_city[i_cat] = city_to_k[desired_city_name]

# 80% thresholds based on Total_recruits (given = 170)
desired_profession_target = int(p2_threshold * Total_recruits)  # 136
desired_city_target = int(p3_threshold * Total_recruits)        # 136

# =========================
# 3. Create model
# =========================
model = gp.Model("Rainbow_Group_Goal_Programming")

# =========================
# 4. Decision variables
# =========================

# x_{ijk}: number of people from category i assigned to major j in city k
x = model.addVars(
    range(1, i_max + 1),
    range(1, j_max + 1),
    range(1, k_max + 1),
    vtype=GRB.INTEGER,
    name="x"
)

# Deviation variables for goals 1, 2, 3
d1_plus = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="d1_plus")
d1_minus = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="d1_minus")

d2_plus = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="d2_plus")
d2_minus = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="d2_minus")

d3_plus = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="d3_plus")
d3_minus = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="d3_minus")

# Upper bounds on x variables by category supply (<=30 each category)
for i in range(1, i_max + 1):
    for j in range(1, j_max + 1):
        for k in range(1, k_max + 1):
            x[i, j, k].ub = category_supply[i]

# =========================
# 5. Objective function
# =========================
# Z = p1(d1+ + d1-) + p2(d2+ + d2-) + p3(d3+ + d3-)
model.setObjective(
    p1 * (d1_plus + d1_minus)
    + p2 * (d2_plus + d2_minus)
    + p3 * (d3_plus + d3_minus),
    GRB.MINIMIZE
)

# =========================
# 6. Constraints
# =========================

# (1) Total recruitment: sum x_ijk = Total_recruits
model.addConstr(
    gp.quicksum(x[i, j, k] for i in range(1, i_max + 1)
                for j in range(1, j_max + 1)
                for k in range(1, k_max + 1)) == Total_recruits,
    name="Total_Recruitment"
)

# (2) City-major quotas from Table_1_C_24
for (j, k), q in quota.items():
    model.addConstr(
        gp.quicksum(x[i, j, k] for i in range(1, i_max + 1)) == q,
        name=f"Quota_j{j}_k{k}"
    )

# (3) Supply limit per category: sum_{j,k} x_{ijk} <= category_supply[i]
for i in range(1, i_max + 1):
    model.addConstr(
        gp.quicksum(x[i, j, k] for j in range(1, j_max + 1)
                    for k in range(1, k_max + 1)) <= category_supply[i],
        name=f"Supply_Limit_Category_{i}"
    )

# (4) Goal 1: Suitability
# sum_{i} sum_{j notin S_i} sum_{k} x_{ijk} + d1^- - d1^+ = 0
unsuitable_expr = gp.LinExpr()
for i in range(1, i_max + 1):
    suitable_set = category_suitable[i]
    for j in range(1, j_max + 1):
        if j not in suitable_set:
            for k in range(1, k_max + 1):
                unsuitable_expr += x[i, j, k]

model.addConstr(
    unsuitable_expr + d1_minus - d1_plus == 0,
    name="Goal1_Suitability"
)

# (5) Goal 2: Desired profession
# sum_{i} sum_{k} x_{i, j_desired(i), k} + d2^- - d2^+ = desired_profession_target
desired_prof_expr = gp.LinExpr()
for i in range(1, i_max + 1):
    j_des = category_desired_major[i]
    for k in range(1, k_max + 1):
        desired_prof_expr += x[i, j_des, k]

model.addConstr(
    desired_prof_expr + d2_minus - d2_plus == desired_profession_target,
    name="Goal2_DesiredProfession"
)

# (6) Goal 3: Desired city
# sum_{i} sum_{j} x_{i, j, k_desired(i)} + d3^- - d3^+ = desired_city_target
desired_city_expr = gp.LinExpr()
for i in range(1, i_max + 1):
    k_des = category_desired_city[i]
    for j in range(1, j_max + 1):
        desired_city_expr += x[i, j, k_des]

model.addConstr(
    desired_city_expr + d3_minus - d3_plus == desired_city_target,
    name="Goal3_DesiredCity"
)

# =========================
# 7. Optimize model
# =========================
model.Params.OutputFlag = 0  # turn off solver log; set to 1 if you want to see details
model.optimize()

if model.Status != GRB.OPTIMAL:
    raise RuntimeError(f"Optimization ended with status {model.Status}")

# =========================
# 8. Extract and print results
# =========================

# Number of people in category 1, desired major Production (j=1), city A (k=1)
x_1_1_1_value = int(round(x[1, 1, 1].X))

print("Optimal objective value:", model.ObjVal)
print(f"x[1,1,1] (Category 1, Production, City A) = {x_1_1_1_value}")

# Final required answer output
print(f"FinalAnswer=【{x_1_1_1_value}】")