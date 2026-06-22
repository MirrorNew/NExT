import gurobipy as gp
from gurobipy import GRB
import math

# 1. Define Data and Parameters
cities_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
plans = ['A', 'B', 'C']
ages = ['0-6', '6-12', '12-18', '18-35', '35-65', '65-100']

# Startup costs
startup_costs = {'A': 50, 'B': 55, 'C': 40}

# City Populations
city_population = {
    'A': 600, 'B': 650, 'C': 620, 'D': 600, 'E': 560, 
    'F': 540, 'G': 590, 'H': 580, 'I': 610, 'J': 500
}

# Raw Premium Data provided in parameters
# Format: [Plan, Age, Premium, Deductible, DocFees, SpecFees]
raw_premium_data = [
    ['A', '0-6', 200, 900, 30, 70],
    ['A', '6-12', 200, 950, 28, 68],
    ['A', '12-18', 200, 1000, 28, 70],
    [None, None, 200, 1000, 35, 70],  # Context: Plan A, 18-35
    ['A', '65-100', 200, 1100, 35, 75],
    ['B', '0-6', 180, 1150, 38, 80],
    ['B', '6-12', 180, 1180, 39, 75],
    ['B', '12-18', 180, 1200, 40, 75],
    ['B', '18-35', 180, 1200, 40, 80],
    ['B', '35-65', 180, 1200, 40, 85],
    [None, None, 220, 750, 25, 65],   # Context: Plan C, 0-6
    ['C', '6-12', 220, 780, 22, 63],
    ['C', '12-18', 220, 800, 20, 60],
    [None, '35-65', 220, 850, 22, 63], # Context: Plan C, 35-65
    ['C', '65-100', 220, 875, 25, 65]
]

# Process and Map Data to (Plan, Age) keys
# We manually map based on the sequence and gaps identified
# Missing from raw: (A, 35-65), (B, 65-100), (C, 18-35)
# Strategy: Use adjacent age group data for missing entries to maintain continuity.

# Helper to store: { (Plan, Age) : { 'Prem': .., 'Ded': .., 'Doc': .., 'Spec': .. } }
cost_data = {}

# Map the 15 raw rows explicitly based on list order
# Plan A
cost_data[('A', '0-6')]    = {'Prem': 200, 'Ded': 900,  'Doc': 30, 'Spec': 70}
cost_data[('A', '6-12')]   = {'Prem': 200, 'Ded': 950,  'Doc': 28, 'Spec': 68}
cost_data[('A', '12-18')]  = {'Prem': 200, 'Ded': 1000, 'Doc': 28, 'Spec': 70}
cost_data[('A', '18-35')]  = {'Prem': 200, 'Ded': 1000, 'Doc': 35, 'Spec': 70} # Row 3
cost_data[('A', '65-100')] = {'Prem': 200, 'Ded': 1100, 'Doc': 35, 'Spec': 75}
# Fill A missing: 35-65 (Replicate 18-35)
cost_data[('A', '35-65')]  = {'Prem': 200, 'Ded': 1000, 'Doc': 35, 'Spec': 70}

# Plan B
cost_data[('B', '0-6')]    = {'Prem': 180, 'Ded': 1150, 'Doc': 38, 'Spec': 80}
cost_data[('B', '6-12')]   = {'Prem': 180, 'Ded': 1180, 'Doc': 39, 'Spec': 75}
cost_data[('B', '12-18')]  = {'Prem': 180, 'Ded': 1200, 'Doc': 40, 'Spec': 75}
cost_data[('B', '18-35')]  = {'Prem': 180, 'Ded': 1200, 'Doc': 40, 'Spec': 80}
cost_data[('B', '35-65')]  = {'Prem': 180, 'Ded': 1200, 'Doc': 40, 'Spec': 85}
# Fill B missing: 65-100 (Replicate 35-65)
cost_data[('B', '65-100')] = {'Prem': 180, 'Ded': 1200, 'Doc': 40, 'Spec': 85}

# Plan C
cost_data[('C', '0-6')]    = {'Prem': 220, 'Ded': 750,  'Doc': 25, 'Spec': 65} # Row 10
cost_data[('C', '6-12')]   = {'Prem': 220, 'Ded': 780,  'Doc': 22, 'Spec': 63}
cost_data[('C', '12-18')]  = {'Prem': 220, 'Ded': 800,  'Doc': 20, 'Spec': 60}
cost_data[('C', '35-65')]  = {'Prem': 220, 'Ded': 850,  'Doc': 22, 'Spec': 63} # Row 13
cost_data[('C', '65-100')] = {'Prem': 220, 'Ded': 875,  'Doc': 25, 'Spec': 65}
# Fill C missing: 18-35 (Replicate 12-18)
cost_data[('C', '18-35')]  = {'Prem': 220, 'Ded': 800,  'Doc': 20, 'Spec': 60}

# 2. Create Model
model = gp.Model("InsurancePolicyOptimization")

# 3. Decision Variables
# n[i, p, a]: Number of customers in city i, plan p, age group a
n = model.addVars(cities_list, plans, ages, vtype=GRB.INTEGER, name="n")

# y[p]: Binary indicator for plan usage (1 if plan p is used, 0 otherwise)
y = model.addVars(plans, vtype=GRB.BINARY, name="y")

# 4. Objective Function
# Minimize sum of deductibles + doctor fees + specialist fees + startup costs
# Variable cost per customer = Ded + Doc + Spec
obj_expr = gp.quicksum(
    (cost_data[(p, a)]['Ded'] + cost_data[(p, a)]['Doc'] + cost_data[(p, a)]['Spec']) * n[i, p, a]
    for i in cities_list for p in plans for a in ages
)
startup_expr = gp.quicksum(startup_costs[p] * y[p] for p in plans)

model.setObjective(obj_expr + startup_expr, GRB.MINIMIZE)

# 5. Constraints

# (1) Demand satisfaction per city
for i in cities_list:
    model.addConstr(
        gp.quicksum(n[i, p, a] for p in plans for a in ages) == city_population[i],
        name=f"Demand_City_{i}"
    )

# (2) Total Period Demand (Implicitly satisfied by (1), but checking total sums match)
# No extra constraint needed if (1) holds for all i.

# (3) Average Deductible Limit
# Sum(Ded * n) <= 1100 * Total_Population
total_pop = sum(city_population.values())
model.addConstr(
    gp.quicksum(cost_data[(p, a)]['Ded'] * n[i, p, a] for i in cities_list for p in plans for a in ages)
    <= 1100 * total_pop,
    name="Max_Avg_Deductible"
)

# (4) Premium Income Requirement
# Sum(Prem * n) >= 1,000,000
model.addConstr(
    gp.quicksum(cost_data[(p, a)]['Prem'] * n[i, p, a] for i in cities_list for p in plans for a in ages)
    >= 1000000,
    name="Min_Premium_Income"
)

# (5) Age Penetration Requirements (18-35 age group)
age_18_35 = '18-35'
# Plan A: >= 25%
model.addConstr(
    gp.quicksum(n[i, 'A', age_18_35] for i in cities_list) 
    >= 0.25 * gp.quicksum(n[i, 'A', a] for i in cities_list for a in ages),
    name="Penetration_A_18_35"
)
# Plan B: >= 30%
model.addConstr(
    gp.quicksum(n[i, 'B', age_18_35] for i in cities_list) 
    >= 0.30 * gp.quicksum(n[i, 'B', a] for i in cities_list for a in ages),
    name="Penetration_B_18_35"
)
# Plan C: >= 20%
model.addConstr(
    gp.quicksum(n[i, 'C', age_18_35] for i in cities_list) 
    >= 0.20 * gp.quicksum(n[i, 'C', a] for i in cities_list for a in ages),
    name="Penetration_C_18_35"
)

# (6) Minimum Sub-quota
# n[i,p,a] >= floor(Pop_i / 60)
for i in cities_list:
    min_quota = math.floor(city_population[i] / 60)
    for p in plans:
        for a in ages:
            model.addConstr(n[i, p, a] >= min_quota, name=f"Quota_{i}_{p}_{a}")

# (7) Plan Activation Linking (Indicator Constraints)
# If sum(n) == 0 -> y = 0.
# (Note: Since minimization objective puts cost on y=1, and n>=min_quota>0 forces y=1, 
# strictly we need y=0 if inactive. The sub-quota actually forces y=1 here.)
for p in plans:
    total_customers_p = gp.quicksum(n[i, p, a] for i in cities_list for a in ages)
    model.addGenConstrIndicator(y[p], 0, total_customers_p == 0, name=f"Indicator_Off_{p}")
    # Optional reverse for completeness in general case, though objective handles it:
    # model.addGenConstrIndicator(y[p], 1, total_customers_p >= 1)

# 6. Solve
model.optimize()

# 7. Print Results
if model.status == GRB.OPTIMAL:
    print("Optimal Solution Found.")
    print(f"Objective Value: {model.objVal}")
    # for p in plans:
    #     print(f"Plan {p} Used: {y[p].X}")
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found.")