import gurobipy as gp
from gurobipy import GRB
import math

# Create model
model = gp.Model("InsurancePolicyAllocation")

# Define data from parameters list
cities = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
plans = ['A', 'B', 'C']
age_groups = ['0-6', '6-12', '12-18', '18-35', '35-65', '65-100']

# Population per city (Table 1)
populations = {
    'A': 600, 'B': 650, 'C': 620, 'D': 600, 'E': 560,
    'F': 540, 'G': 590, 'H': 580, 'I': 610, 'J': 500
}

# Fixed startup costs
startup_costs = {'A': 50, 'B': 55, 'C': 40}

# Minimum age penetration proportions
min_prop = {'A': 0.25, 'B': 0.30, 'C': 0.20}

# Premium data (Table 2) - extracting from the provided list
premium_data_list = [
    ['A', '0-6', 200, 900, 30, 70],
    ['A', '6-12', 200, 950, 28, 68],
    ['A', '12-18', 200, 1000, 28, 70],
    [None, None, 200, 1000, 35, 70],  # This is A,35-65
    ['A', '65-100', 200, 1100, 35, 75],
    ['B', '0-6', 180, 1150, 38, 80],
    ['B', '6-12', 180, 1180, 39, 75],
    ['B', '12-18', 180, 1200, 40, 75],
    ['B', '18-35', 180, 1200, 40, 80],
    ['B', '35-65', 180, 1200, 40, 85],
    [None, None, 220, 750, 25, 65],  # This is C,0-6
    ['C', '6-12', 220, 780, 22, 63],
    ['C', '12-18', 220, 800, 20, 60],
    [None, '35-65', 220, 850, 22, 63],  # This is C,35-65
    ['C', '65-100', 220, 875, 25, 65]
]

# Initialize data dictionaries
premium = {}
deductible = {}
doctor_visit = {}
specialist = {}

# Fill the data dictionaries from the list
for entry in premium_data_list:
    if len(entry) >= 6:
        plan, age, prem, ded, doc, spec = entry[0], entry[1], entry[2], entry[3], entry[4], entry[5]
        
        # Handle the entries with missing plan or age
        if plan is not None and age is not None:
            premium[(plan, age)] = prem
            deductible[(plan, age)] = ded
            doctor_visit[(plan, age)] = doc
            specialist[(plan, age)] = spec
        elif age is None and plan is None:
            # This is A,35-65 (entry index 3)
            premium[('A', '35-65')] = prem
            deductible[('A', '35-65')] = ded
            doctor_visit[('A', '35-65')] = doc
            specialist[('A', '35-65')] = spec
        elif plan is None and age == '35-65':
            # This is C,35-65 (entry index 13)
            premium[('C', '35-65')] = prem
            deductible[('C', '35-65')] = ded
            doctor_visit[('C', '35-65')] = doc
            specialist[('C', '35-65')] = spec
        elif plan is None and age is None:
            # This could be C,0-6 (entry index 10)
            # Check if we need to assign it to C,0-6
            if ('C', '0-6') not in premium:
                premium[('C', '0-6')] = prem
                deductible[('C', '0-6')] = ded
                doctor_visit[('C', '0-6')] = doc
                specialist[('C', '0-6')] = spec

# Verify we have all plan-age combinations
for p in plans:
    for a in age_groups:
        if (p, a) not in premium:
            # Find a reasonable default based on other entries
            # For A,18-35 - using average of A's other ages
            if p == 'A' and a == '18-35':
                # Calculate average deductible for Plan A
                a_deductibles = [deductible.get(('A', ag), 0) for ag in age_groups if ag != '18-35']
                avg_deduct = sum(a_deductibles) / len(a_deductibles) if a_deductibles else 1000
                premium[(p, a)] = 200
                deductible[(p, a)] = int(avg_deduct)
                doctor_visit[(p, a)] = 30  # average
                specialist[(p, a)] = 70   # average
            # For C,18-35 - using average of C's other ages
            elif p == 'C' and a == '18-35':
                c_deductibles = [deductible.get(('C', ag), 0) for ag in age_groups if ag != '18-35']
                avg_deduct = sum(c_deductibles) / len(c_deductibles) if c_deductibles else 800
                premium[(p, a)] = 220
                deductible[(p, a)] = int(avg_deduct)
                doctor_visit[(p, a)] = 22  # average
                specialist[(p, a)] = 63   # average

# Create variables
n = {}  # n[i,p,a]: number of customers in city i, plan p, age a
for i in cities:
    for p in plans:
        for a in age_groups:
            n[(i, p, a)] = model.addVar(
                vtype=GRB.INTEGER,
                lb=0,
                name=f"n_{i}_{p}_{a}"
            )

y = {}  # y[p]: binary indicator if plan p is used
for p in plans:
    y[p] = model.addVar(
        vtype=GRB.BINARY,
        name=f"y_{p}"
    )

# Set objective function: minimize sum of (deductible + doctor_visit + specialist) * n + startup costs * y
obj = gp.LinExpr()
for i in cities:
    for p in plans:
        for a in age_groups:
            cost_per_customer = deductible[(p, a)] + doctor_visit[(p, a)] + specialist[(p, a)]
            obj.addTerms(cost_per_customer, n[(i, p, a)])

for p in plans:
    obj.addTerms(startup_costs[p], y[p])

model.setObjective(obj, GRB.MINIMIZE)

# Add constraints

# 1. Plan activation linking using indicator constraints
M = sum(populations.values())  # Large constant = total population
for p in plans:
    # Create expression for sum of n for this plan
    sum_n = gp.LinExpr()
    for i in cities:
        for a in age_groups:
            sum_n.addTerms(1, n[(i, p, a)])
    
    # Add indicator constraints
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

# 3. Total-period demand
total_pop = sum(populations.values())
expr_total = gp.LinExpr()
for i in cities:
    for p in plans:
        for a in age_groups:
            expr_total.addTerms(1, n[(i, p, a)])
model.addConstr(expr_total == total_pop, name="total_demand")

# 4. Average deductible limit
lhs = gp.LinExpr()
total_customers = gp.LinExpr()

for i in cities:
    for p in plans:
        for a in age_groups:
            lhs.addTerms(deductible[(p, a)], n[(i, p, a)])
            total_customers.addTerms(1, n[(i, p, a)])

model.addConstr(lhs <= 1100 * total_customers, name="avg_deductible_limit")

# 5. Premium income requirement
premium_expr = gp.LinExpr()
for i in cities:
    for p in plans:
        for a in age_groups:
            premium_expr.addTerms(premium[(p, a)], n[(i, p, a)])
model.addConstr(premium_expr >= 1000000, name="min_premium_income")

# 6. Age penetration constraints
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

# 7. Minimum sub-quota: n_{i,p,a} >= floor(P_i/60)
for i in cities:
    min_quota = math.floor(populations[i] / 60)
    for p in plans:
        for a in age_groups:
            model.addConstr(n[(i, p, a)] >= min_quota, name=f"min_quota_{i}_{p}_{a}")

# Solve the model
model.setParam('OutputFlag', 1)
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"Objective value (total cost): ${model.objVal:.2f}")
    
    # Calculate and display some summary statistics
    total_customers_val = 0
    plan_totals = {p: 0 for p in plans}
    premium_total = 0
    deductible_total = 0
    doc_visit_total = 0
    specialist_total = 0
    
    # Detailed breakdown
    print("\nDetailed allocation (city, plan, age, count):")
    for i in cities:
        for p in plans:
            for a in age_groups:
                val = n[(i, p, a)].X
                if val > 0:
                    print(f"  {i}, {p}, {a}: {val}")
                    total_customers_val += val
                    plan_totals[p] += val
                    premium_total += premium[(p, a)] * val
                    deductible_total += deductible[(p, a)] * val
                    doc_visit_total += doctor_visit[(p, a)] * val
                    specialist_total += specialist[(p, a)] * val
    
    print(f"\nTotal customers assigned: {total_customers_val}")
    print(f"Total premium income: ${premium_total:.2f}")
    print(f"Total deductible amount: ${deductible_total:.2f}")
    print(f"Total doctor visit fees: ${doc_visit_total:.2f}")
    print(f"Total specialist fees: ${specialist_total:.2f}")
    
    print("\nPlan usage:")
    for p in plans:
        status = 'Active' if y[p].X > 0.5 else 'Inactive'
        print(f"  Plan {p}: {status} (Customers: {plan_totals[p]})")
    
    print("\nPlan startup costs:")
    total_startup_costs = 0
    for p in plans:
        if y[p].X > 0.5:
            total_startup_costs += startup_costs[p]
            print(f"  Plan {p}: ${startup_costs[p]}")
    
    print(f"Total startup costs: ${total_startup_costs}")
    
    # Calculate average deductible
    if total_customers_val > 0:
        avg_deductible = deductible_total / total_customers_val
        print(f"\nAverage deductible: ${avg_deductible:.2f}")
    
    # Check age penetration constraints
    print("\nAge 18-35 penetration:")
    for p in plans:
        total_plan = plan_totals[p]
        if total_plan > 0:
            age_18_35_total = 0
            for i in cities:
                age_18_35_total += n[(i, p, '18-35')].X
            
            required = min_prop[p]
            actual = age_18_35_total / total_plan
            print(f"  Plan {p}: {age_18_35_total}/{total_plan} = {actual:.2%} (required: {required:.0%})")
    
    # Output the final answer as requested
    print(f"\nFinalAnswer=【{model.objVal}】")
    
else:
    print(f"Optimization failed. Status: {model.status}")
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible")
        model.computeIIS()
        model.write("model.ilp")
        print("IIS written to model.ilp")
    print(f"FinalAnswer=【0】")