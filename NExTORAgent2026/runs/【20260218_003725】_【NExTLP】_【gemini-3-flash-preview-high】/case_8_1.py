import gurobipy as gp
from gurobipy import GRB
import math

# 1. Define sets and data
cities = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
plans = ['A', 'B', 'C']
age_groups = ['0-6', '6-12', '12-18', '18-35', '35-65', '65-100']

# City populations
city_pop = {'A': 600, 'B': 650, 'C': 620, 'D': 600, 'E': 560, 'F': 540, 'G': 590, 'H': 580, 'I': 610, 'J': 500}

# Startup costs for each plan
startup_costs = {'A': 50, 'B': 55, 'C': 40}

# Requirement parameters
max_avg_deductible = 1100
min_total_premium_income = 1000000
min_prop_18_35 = {'A': 0.25, 'B': 0.3, 'C': 0.2}

# Policy-specific data (Premium, Deductible, Doctor Fee, Specialist Fee)
# Combined information from Table_2_PremiumData and the math model description
policy_data = {
    ('A', '0-6'):     (200, 900, 30, 70),
    ('A', '6-12'):    (200, 950, 28, 68),
    ('A', '12-18'):   (200, 1000, 28, 70),
    ('A', '18-35'):   (200, 1000, 35, 70),
    ('A', '35-65'):   (200, 1050, 35, 72), # From math model context
    ('A', '65-100'):  (200, 1100, 35, 75),
    
    ('B', '0-6'):     (180, 1150, 38, 80),
    ('B', '6-12'):    (180, 1180, 39, 75),
    ('B', '12-18'):   (180, 1200, 40, 75),
    ('B', '18-35'):   (180, 1200, 40, 80),
    ('B', '35-65'):   (180, 1200, 40, 85),
    ('B', '65-100'):  (180, 1250, 42, 90), # From math model context
    
    ('C', '0-6'):     (220, 750, 25, 65),
    ('C', '6-12'):    (220, 780, 22, 63),
    ('C', '12-18'):   (220, 800, 20, 60),
    ('C', '18-35'):   (220, 825, 21, 62), # From math model context
    ('C', '35-65'):   (220, 850, 22, 63),
    ('C', '65-100'):  (220, 875, 25, 65)
}

# 2. Create the model
model = gp.Model("InsurancePolicyAllocation")

# 3. Create decision variables
# n[i, p, a] is number of customers in city i assigned to plan p and age group a
n = model.addVars(cities, plans, age_groups, vtype=GRB.INTEGER, lb=0, name="n")
# y[p] is binary, 1 if plan p is used
y = model.addVars(plans, vtype=GRB.BINARY, name="y")

# 4. Set up the objective function
# Min Z = sum of (deductible + doctor fee + specialist fee) * n + sum of startup_costs * y
total_variable_cost = gp.quicksum(
    n[i, p, a] * (policy_data[p, a][1] + policy_data[p, a][2] + policy_data[p, a][3])
    for i in cities for p in plans for a in age_groups
)
total_startup_cost = gp.quicksum(startup_costs[p] * y[p] for p in plans)
model.setObjective(total_variable_cost + total_startup_cost, GRB.MINIMIZE)

# 5. Add constraints

# Demand satisfaction per city
for i in cities:
    model.addConstr(gp.quicksum(n[i, p, a] for p in plans for a in age_groups) == city_pop[i], name=f"Demand_{i}")

# Minimum sub-quota constraint: n_{i,p,a} >= floor(P_i / 60)
for i in cities:
    min_quota = math.floor(city_pop[i] / 60)
    for p in plans:
        for a in age_groups:
            model.addConstr(n[i, p, a] >= min_quota, name=f"Quota_{i}_{p}_{a}")

# Plan-activation indicator constraints
for p in plans:
    plan_usage = gp.quicksum(n[i, p, a] for i in cities for a in age_groups)
    # y_p = 1 if plan p has customers
    model.addGenConstrIndicator(y[p], 1, plan_usage >= 1)
    # y_p = 0 if plan p has zero customers
    model.addGenConstrIndicator(y[p], 0, plan_usage == 0)

# Average deductible limit
total_deductible = gp.quicksum(n[i, p, a] * policy_data[p, a][1] for i in cities for p in plans for a in age_groups)
total_pop = sum(city_pop.values())
model.addConstr(total_deductible <= max_avg_deductible * total_pop, name="AvgDeductible")

# Premium income requirement
total_premium = gp.quicksum(n[i, p, a] * policy_data[p, a][0] for i in cities for p in plans for a in age_groups)
model.addConstr(total_premium >= min_total_premium_income, name="MinPremium")

# Age penetration Plan A, B, C for 18-35 group
for p in plans:
    n_p_18_35 = gp.quicksum(n[i, p, '18-35'] for i in cities)
    n_p_total = gp.quicksum(n[i, p, a] for i in cities for a in age_groups)
    model.addConstr(n_p_18_35 >= min_prop_18_35[p] * n_p_total, name=f"Penetration_{p}")

# 6. Solve the model
model.optimize()

# 7. Print results
if model.status == GRB.OPTIMAL:
    print(f"Optimal Total Cost: {model.objVal}")
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("Optimal solution not found.")