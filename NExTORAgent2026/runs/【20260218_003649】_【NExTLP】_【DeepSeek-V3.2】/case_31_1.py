import gurobipy as gp
from gurobipy import GRB

# Data from parameters list
Total_recruits = 170
i_max = 6
j_max = 3
k_max = 2
p1 = 10000
p2 = 100
p3 = 1
p2_threshold = 0.8
p3_threshold = 0.8

# Category information from Table_2_C_25
Table_2_C_25 = [
    [1, 30, ['Production', 'Marketing'], 'Production', 'A'],
    [2, 30, ['Marketing', 'Finance'], 'Marketing', 'A'],
    [3, 30, ['Production', 'Finance'], 'Production', 'B'],
    [4, 30, ['Production', 'Finance'], 'Finance', 'B'],
    [5, 30, ['Marketing', 'Finance'], 'Finance', 'A'],
    [6, 30, ['Finance'], 'Finance', 'B']
]

# Create model
model = gp.Model("Rainbow_Group_Recruitment")

# Decision variables x_{ijk}
x = {}
for i in range(1, i_max + 1):
    for j in range(1, j_max + 1):
        for k in range(1, k_max + 1):
            x[i, j, k] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=30, name=f"x_{i}_{j}_{k}")

# Deviation variables
d1_plus = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="d1_plus")
d1_minus = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="d1_minus")
d2_plus = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="d2_plus")
d2_minus = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="d2_minus")
d3_plus = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="d3_plus")
d3_minus = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="d3_minus")

# Update model to incorporate variables
model.update()

# Constraint 1: Total recruitment = 170
model.addConstr(
    gp.quicksum(x[i, j, k] for i in range(1, i_max + 1)
                for j in range(1, j_max + 1)
                for k in range(1, k_max + 1)) == Total_recruits,
    name="Total_Recruitment"
)

# Constraints 2: City-Profession quotas from Table_1_C_24
# A, Production = 20
model.addConstr(
    gp.quicksum(x[i, 1, 1] for i in range(1, i_max + 1)) == 20,
    name="Quota_A_Production"
)
# A, Marketing = 30
model.addConstr(
    gp.quicksum(x[i, 2, 1] for i in range(1, i_max + 1)) == 30,
    name="Quota_A_Marketing"
)
# A, Finance = 40
model.addConstr(
    gp.quicksum(x[i, 3, 1] for i in range(1, i_max + 1)) == 40,
    name="Quota_A_Finance"
)
# B, Production = 25
model.addConstr(
    gp.quicksum(x[i, 1, 2] for i in range(1, i_max + 1)) == 25,
    name="Quota_B_Production"
)
# B, Marketing = 20
model.addConstr(
    gp.quicksum(x[i, 2, 2] for i in range(1, i_max + 1)) == 20,
    name="Quota_B_Marketing"
)
# B, Finance = 35
model.addConstr(
    gp.quicksum(x[i, 3, 2] for i in range(1, i_max + 1)) == 35,
    name="Quota_B_Finance"
)

# Constraints 3: Supply limits per category ≤ 30
for i in range(1, i_max + 1):
    model.addConstr(
        gp.quicksum(x[i, j, k] for j in range(1, j_max + 1)
                    for k in range(1, k_max + 1)) <= 30,
        name=f"Supply_Limit_Category_{i}"
    )

# Goal 1: Suitable major constraint
# Define S_i: set of suitable majors for each category i
suitable_sets = {
    1: [1, 2],   # Production, Marketing
    2: [2, 3],   # Marketing, Finance
    3: [1, 3],   # Production, Finance
    4: [1, 3],   # Production, Finance
    5: [2, 3],   # Marketing, Finance
    6: [3]       # Finance
}

# Sum over all unsuitable assignments
unsuitable_sum = gp.QuadExpr()
for i in range(1, i_max + 1):
    suitable = suitable_sets[i]
    for j in range(1, j_max + 1):
        if j not in suitable:
            for k in range(1, k_max + 1):
                unsuitable_sum += x[i, j, k]

model.addConstr(
    unsuitable_sum + d1_minus - d1_plus == 0,
    name="Goal1_Suitability"
)

# Goal 2: Desired profession (≥ 80% of 170 = 136)
# Define desired profession for each category
desired_profession = {
    1: 1,  # Production
    2: 2,  # Marketing
    3: 1,  # Production
    4: 3,  # Finance
    5: 3,  # Finance
    6: 3   # Finance
}

desired_prof_sum = gp.QuadExpr()
for i in range(1, i_max + 1):
    j_des = desired_profession[i]
    for k in range(1, k_max + 1):
        desired_prof_sum += x[i, j_des, k]

model.addConstr(
    desired_prof_sum + d2_minus - d2_plus == 136,
    name="Goal2_DesiredProfession"
)

# Goal 3: Desired city (≥ 80% of 170 = 136)
# Define desired city for each category
desired_city = {
    1: 1,  # A
    2: 1,  # A
    3: 2,  # B
    4: 2,  # B
    5: 1,  # A
    6: 2   # B
}

desired_city_sum = gp.QuadExpr()
for i in range(1, i_max + 1):
    k_des = desired_city[i]
    for j in range(1, j_max + 1):
        desired_city_sum += x[i, j, k_des]

model.addConstr(
    desired_city_sum + d3_minus - d3_plus == 136,
    name="Goal3_DesiredCity"
)

# Objective function: minimize weighted deviations
model.setObjective(
    p1 * (d1_plus + d1_minus) +
    p2 * (d2_plus + d2_minus) +
    p3 * (d3_plus + d3_minus),
    GRB.MINIMIZE
)

# Solve the model
model.optimize()

# Check if optimal solution found
if model.status == GRB.OPTIMAL:
    # Get the value for category 1, production (j=1), city A (k=1)
    answer = x[1, 1, 1].X
    print(f"FinalAnswer=【{int(answer)}】")
else:
    print("Model did not solve to optimality.")
    print(f"FinalAnswer=【0】")