import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs
# City/Profession Quotas from Table C-24
# Cities: 0: A, 1: B
# Majors: 0: Production, 1: Marketing, 2: Finance
quotas = {
    (0, 0): 20, # A, Production
    (1, 0): 30, # A, Marketing
    (2, 0): 40, # A, Finance
    (0, 1): 25, # B, Production
    (1, 1): 20, # B, Marketing
    (2, 1): 35  # B, Finance
}

# Category data from Table C-25
# (Category ID, Count, Suitable Majors, Desired Major, Desired City)
# Using 0-indexing for majors (P:0, M:1, F:2) and cities (A:0, B:1)
categories = [
    (0, 30, [0, 1], 0, 0), # Cat 1
    (1, 30, [1, 2], 1, 0), # Cat 2
    (2, 30, [0, 2], 0, 1), # Cat 3
    (3, 30, [0, 2], 2, 1), # Cat 4
    (4, 30, [1, 2], 2, 0), # Cat 5
    (5, 30, [2], 2, 1)     # Cat 6
]

# Weights for Goal Programming
p1_weight = 10000
p2_weight = 100
p3_weight = 1

# Goal targets
total_recruits_target = 170
goal_2_3_threshold = int(total_recruits_target * 0.8) # 80% of 170 = 136

# 2. Create the Gurobi Model
model = gp.Model("RainbowGroupRecruitment")

# 3. Create decision variables
# x[i, j, k]: number of people from category i assigned to major j in city k
x = model.addVars(6, 3, 2, vtype=GRB.INTEGER, lb=0, ub=30, name="x")

# Deviation variables for goal programming
d1p = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="d1p")
d1m = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="d1m")
d2p = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="d2p")
d2m = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="d2m")
d3p = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="d3p")
d3m = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="d3m")

# 4. Set up the objective function
# Minimize the weighted sum of deviations
model.setObjective(p1_weight * (d1p + d1m) + p2_weight * (d2p + d2m) + p3_weight * (d3p + d3m), GRB.MINIMIZE)

# 5. Add all constraints
# Hard Constraint: Total Recruitment
model.addConstr(x.sum() == total_recruits_target, name="Total_Recruitment")

# Hard Constraint: Quotas for each city and profession
for (j, k), quota in quotas.items():
    model.addConstr(gp.quicksum(x[i, j, k] for i in range(6)) == quota, name=f"Quota_{k}_{j}")

# Hard Constraint: Supply Limit per Category
for i in range(6):
    model.addConstr(gp.quicksum(x[i, j, k] for j in range(3) for k in range(2)) <= 30, name=f"Supply_Limit_{i}")

# Goal 1: Suitability (Minimize unsuitable assignments)
# Identify indices (i, j) that are NOT suitable
unsuitable_expr = gp.LinExpr()
for i, count, suitable_majors, desired_major, desired_city in categories:
    for j in range(3):
        if j not in suitable_majors:
            for k in range(2):
                unsuitable_expr += x[i, j, k]
model.addConstr(unsuitable_expr + d1m - d1p == 0, name="Goal1_Suitability")

# Goal 2: Desired Profession (Engaged in the profession they hope for)
desired_prof_expr = gp.LinExpr()
for i, count, suitable_majors, desired_major, desired_city in categories:
    for k in range(2):
        desired_prof_expr += x[i, desired_major, k]
model.addConstr(desired_prof_expr + d2m - d2p == goal_2_3_threshold, name="Goal2_DesiredProfession")

# Goal 3: Desired City (Working in the city they hope for)
desired_city_expr = gp.LinExpr()
for i, count, suitable_majors, desired_major, desired_city in categories:
    for j in range(3):
        desired_city_expr += x[i, j, desired_city]
model.addConstr(desired_city_expr + d3m - d3p == goal_2_3_threshold, name="Goal3_DesiredCity")

# 6. Solve the model
model.setParam('OutputFlag', 0)
model.optimize()

# 7. Print results and output the final answer
# Question asks for category 1 (index 0), assigned to production (index 0) and city A (index 0)
final_answer = int(x[0, 0, 0].X)
print(f"FinalAnswer=【{final_answer}】")