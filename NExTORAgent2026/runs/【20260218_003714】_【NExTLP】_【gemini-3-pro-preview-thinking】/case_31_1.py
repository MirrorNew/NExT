import gurobipy as gp
from gurobipy import GRB

# 1. Define Model
model = gp.Model("RainbowGroup_Recruitment")

# 2. Parameters and Data
# Indices
categories = [1, 2, 3, 4, 5, 6]
professions = [1, 2, 3] # 1:Production, 2:Marketing, 3:Finance
cities = [1, 2]         # 1:A, 2:B

# Data based on Table C-24 (Demand)
# demand[city][profession]
demand = {
    1: {1: 20, 2: 30, 3: 40}, # City A
    2: {1: 25, 2: 20, 3: 35}  # City B
}

# Data based on Table C-25 (Supply and Attributes)
# supply limit is 30 for each category
supply_limit = 30

# Unsuitable sets (j not in Suitable)
# Cat 1 (Prod, Mkt) -> Unsuitable: Finance (3)
# Cat 2 (Mkt, Fin)  -> Unsuitable: Production (1)
# Cat 3 (Prod, Fin) -> Unsuitable: Marketing (2)
# Cat 4 (Prod, Fin) -> Unsuitable: Marketing (2)
# Cat 5 (Mkt, Fin)  -> Unsuitable: Production (1)
# Cat 6 (Fin)       -> Unsuitable: Production (1), Marketing (2)
unsuitable_professions = {
    1: [3],
    2: [1],
    3: [2],
    4: [2],
    5: [1],
    6: [1, 2]
}

# Desired Profession (j)
# 1: Prod, 2: Mkt, 3: Prod, 4: Fin, 5: Fin, 6: Fin
# Corrected based on text Table C-25:
# 1: Prod(1)
# 2: Mkt(2)
# 3: Prod(1)
# 4: Fin(3)
# 5: Fin(3)
# 6: Fin(3)
desired_profession = {1: 1, 2: 2, 3: 1, 4: 3, 5: 3, 6: 3}

# Desired City (k)
# 1: A(1), 2: A(1), 3: B(2), 4: B(2), 5: A(1), 6: B(2)
desired_city = {1: 1, 2: 1, 3: 2, 4: 2, 5: 1, 6: 2}

# Goal targets
target_recruits = 170
p2_target = 136 # 0.8 * 170
p3_target = 136 # 0.8 * 170

# 3. Decision Variables
# x[i, j, k]: Num recruits from cat i assigned to prof j in city k
x = model.addVars(categories, professions, cities, vtype=GRB.INTEGER, lb=0, name="x")

# Deviation variables for 3 goals
# d1: Suitability, d2: Desired Prof, d3: Desired City
d1_pos = model.addVar(lb=0, name="d1_pos")
d1_neg = model.addVar(lb=0, name="d1_neg")
d2_pos = model.addVar(lb=0, name="d2_pos")
d2_neg = model.addVar(lb=0, name="d2_neg")
d3_pos = model.addVar(lb=0, name="d3_pos")
d3_neg = model.addVar(lb=0, name="d3_neg")

# 4. Constraints

# 4.1 Demand Constraints (Quotas per City and Profession)
for k in cities:
    for j in professions:
        model.addConstr(gp.quicksum(x[i, j, k] for i in categories) == demand[k][j], 
                        name=f"Demand_City{k}_Prof{j}")

# 4.2 Supply Constraints (Per Category)
for i in categories:
    model.addConstr(gp.quicksum(x[i, j, k] for j in professions for k in cities) <= supply_limit, 
                    name=f"Supply_Cat{i}")

# 4.3 Total Recruitment Check (Implicitly satisfied by Demand, but added for completeness)
model.addConstr(gp.quicksum(x[i, j, k] for i in categories for j in professions for k in cities) == target_recruits,
                name="Total_Recruitment")

# 4.4 Goal 1: Suitability
# Sum of assignments to unsuitable professions
unsuitable_assignments = gp.quicksum(
    x[i, j, k] 
    for i in categories 
    for j in unsuitable_professions[i] 
    for k in cities
)
model.addConstr(unsuitable_assignments + d1_neg - d1_pos == 0, name="Goal1_Suitability")

# 4.5 Goal 2: Desired Profession
# Sum of assignments to desired profession
desired_prof_assignments = gp.quicksum(
    x[i, desired_profession[i], k]
    for i in categories
    for k in cities
)
model.addConstr(desired_prof_assignments + d2_neg - d2_pos == p2_target, name="Goal2_Profession")

# 4.6 Goal 3: Desired City
# Sum of assignments to desired city
desired_city_assignments = gp.quicksum(
    x[i, j, desired_city[i]]
    for i in categories
    for j in professions
)
model.addConstr(desired_city_assignments + d3_neg - d3_pos == p3_target, name="Goal3_City")

# 5. Objective Function
# Minimize weighted sum of deviations
# Weights: P1=10000, P2=100, P3=1
obj_expr = 10000 * (d1_pos + d1_neg) + 100 * (d2_pos + d2_neg) + 1 * (d3_pos + d3_neg)
model.setObjective(obj_expr, GRB.MINIMIZE)

# 6. Solve
model.optimize()

# 7. Results and Answer
if model.status == GRB.OPTIMAL:
    print("Optimization Successful")
    
    # Question: "how many people in category 1 who want to engage in production majors and are assigned to city A?"
    # Category 1 "wish" is Production. Assigned to City A.
    # We check x[1, 1, 1] (Cat 1, Prof 1 (Prod), City 1 (A))
    
    ans_value = x[1, 1, 1].x
    print(f"x_1_1_1 = {ans_value}")
    
    # Additional debug prints to verify constraints
    print(f"Goal 1 Unsuitable Count: {unsuitable_assignments.getValue()}")
    print(f"Goal 2 Desired Prof Count: {desired_prof_assignments.getValue()} (Target {p2_target})")
    print(f"Goal 3 Desired City Count: {desired_city_assignments.getValue()} (Target {p3_target})")

    print(f"FinalAnswer=【{int(ans_value)}】")
else:
    print("No optimal solution found.")