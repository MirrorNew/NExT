import gurobipy as gp
from gurobipy import GRB
import math

# Create model
model = gp.Model("InsurancePolicyAllocation")

# Define data from parameters list
cities = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
plans = ['A', 'B', 'C']
age_groups = ['0-6', '6-12', '12-18', '18-35', '35-65', '65-100']

# Population per city
populations = {
    'A': 600, 'B': 650, 'C': 620, 'D': 600, 'E': 560,
    'F': 540, 'G': 590, 'H': 580, 'I': 610, 'J': 500
}

# Fixed startup costs
startup_costs = {'A': 50, 'B': 55, 'C': 40}

# Minimum age penetration proportions
min_prop = {'A': 0.25, 'B': 0.30, 'C': 0.20}

# Premium data table - complete based on pattern
premium_data = {
    ('A', '0-6'): {'premium': 200, 'deductible': 900, 'doctor': 30, 'specialist': 70},
    ('A', '6-12'): {'premium': 200, 'deductible': 950, 'doctor': 28, 'specialist': 68},
    ('A', '12-18'): {'premium': 200, 'deductible': 1000, 'doctor': 28, 'specialist': 70},
    ('A', '18-35'): {'premium': 200, 'deductible': 1000, 'doctor': 35, 'specialist': 70},
    ('A', '35-65'): {'premium': 200, 'deductible': 1000, 'doctor': 35, 'specialist': 70},
    ('A', '65-100'): {'premium': 200, 'deductible': 1100, 'doctor': 35, 'specialist': 75},
    
    ('B', '0-6'): {'premium': 180, 'deductible': 1150, 'doctor': 38, 'specialist': 80},
    ('B', '6-12'): {'premium': 180, 'deductible': 1180, 'doctor': 39, 'specialist': 75},
    ('B', '12-18'): {'premium': 180, 'deductible': 1200, 'doctor': 40, 'specialist': 75},
    ('B', '18-35'): {'premium': 180, 'deductible': 1200, 'doctor': 40, 'specialist': 80},
    ('B', '35-65'): {'premium': 180, 'deductible': 1200, 'doctor': 40, 'specialist': 85},
    ('B', '65-100'): {'premium': 180, 'deductible': 1200, 'doctor': 40, 'specialist': 85},
    
    ('C', '0-6'): {'premium': 220, 'deductible': 750, 'doctor': 25, 'specialist': 65},
    ('C', '6-12'): {'premium': 220, 'deductible': 780, 'doctor': 22, 'specialist': 63},
    ('C', '12-18'): {'premium': 220, 'deductible': 800, 'doctor': 20, 'specialist': 60},
    ('C', '18-35'): {'premium': 220, 'deductible': 800, 'doctor': 20, 'specialist': 60},
    ('C', '35-65'): {'premium': 220, 'deductible': 850, 'doctor': 22, 'specialist': 63},
    ('C', '65-100'): {'premium': 220, 'deductible': 875, 'doctor': 25, 'specialist': 65}
}

# Create variables
n = {}  # n[i,p,a]
for i in cities:
    for p in plans:
        for a in age_groups:
            n[(i, p, a)] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"n_{i}_{p}_{a}")

y = {}  # y[p]
for p in plans:
    y[p] = model.addVar(vtype=GRB.BINARY, name=f"y_{p}")

# Set objective function
obj = gp.LinExpr()
for i in cities:
    for p in plans:
        for a in age_groups:
            data = premium_data[(p, a)]
            cost = data['deductible'] + data['doctor'] + data['specialist']
            obj.addTerms(cost, n[(i, p, a)])

for p in plans:
    obj.addTerms(startup_costs[p], y[p])

model.setObjective(obj, GRB.MINIMIZE)

# Add constraints

# 1. Plan activation linking using indicator constraints
M = sum(populations.values())
for p in plans:
    sum_n = gp.LinExpr()
    for i in cities:
        for a in age_groups:
            sum_n.addTerms(1, n[(i, p, a)])
    
    # If y_p = 0, then sum_n <= 0
    model.addGenConstrIndicator(y[p], 0, sum_n <= 0)
    # If y_p = 1, then sum_n <= M
    model.addGenConstrIndicator(y[p], 1, sum_n <= M)

# 2. Demand satisfaction per city
for i in cities:
    expr = gp.LinExpr()
    for p in plans:
        for a in age_groups:
            expr.addTerms(1, n[(i, p, a)])
    model.addConstr(expr == populations[i], name=f"demand_city_{i}")

# 3. Average deductible limit
lhs = gp.LinExpr()
total_customers = gp.LinExpr()

for i in cities:
    for p in plans:
        for a in age_groups:
            data = premium_data[(p, a)]
            lhs.addTerms(data['deductible'], n[(i, p, a)])
            total_customers.addTerms(1, n[(i, p, a)])

model.addConstr(lhs <= 1100 * total_customers, name="avg_deductible_limit")

# 4. Premium income requirement
premium_expr = gp.LinExpr()
for i in cities:
    for p in plans:
        for a in age_groups:
            data = premium_data[(p, a)]
            premium_expr.addTerms(data['premium'], n[(i, p, a)])

model.addConstr(premium_expr >= 1000000, name="min_premium_income")

# 5. Age penetration constraints
for p in plans:
    # Total customers for plan p
    total_plan = gp.LinExpr()
    for i in cities:
        for a in age_groups:
            total_plan.addTerms(1, n[(i, p, a)])
    
    # Customers in age group 18-35 for plan p
    age_18_35 = gp.LinExpr()
    for i in cities:
        age_18_35.addTerms(1, n[(i, p, '18-35')])
    
    # Add constraint
    model.addConstr(age_18_35 >= min_prop[p] * total_plan, name=f"age_penetration_{p}")

# 6. Minimum sub-quota
for i in cities:
    min_quota = math.floor(populations[i] / 60)
    for p in plans:
        for a in age_groups:
            model.addConstr(n[(i, p, a)] >= min_quota, name=f"min_quota_{i}_{p}_{a}")

# Solve the model
model.setParam('OutputFlag', 0)
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"Objective value (total cost): ${model.objVal:.2f}")
    
    # Calculate detailed costs
    total_deductible = 0
    total_doctor = 0
    total_specialist = 0
    total_premium = 0
    
    for i in cities:
        for p in plans:
            for a in age_groups:
                val = n[(i, p, a)].X
                if val > 0:
                    data = premium_data[(p, a)]
                    total_deductible += data['deductible'] * val
                    total_doctor += data['doctor'] * val
                    total_specialist += data['specialist'] * val
                    total_premium += data['premium'] * val
    
    print(f"Total deductible costs: ${total_deductible:.2f}")
    print(f"Total doctor visit fees: ${total_doctor:.2f}")
    print(f"Total specialist fees: ${total_specialist:.2f}")
    print(f"Total premium income: ${total_premium:.2f}")
    
    total_startup = 0
    for p in plans:
        if y[p].X > 0.5:
            total_startup += startup_costs[p]
    
    print(f"Total startup costs: ${total_startup:.2f}")
    
    # Output the final answer
    print(f"\nFinalAnswer=【{model.objVal}】")
    
else:
    print(f"Optimization failed. Status: {model.status}")
    print(f"FinalAnswer=【0】")