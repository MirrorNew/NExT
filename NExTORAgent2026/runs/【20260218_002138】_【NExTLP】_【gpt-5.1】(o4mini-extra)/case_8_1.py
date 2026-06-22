import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters and Data
# =========================

# From Parameters List (use exactly as given)
factories = ['A', 'B', 'C']
number_of_factories = 3
policy_types = ['A', 'B', 'C']
number_of_policy_types = 3
startup_costs = {'A': 50, 'B': 55, 'C': 40}
num_customers_per_city = 600
num_cities = 10
max_average_deductible = 1100
min_total_premium_income = 1000000
age_groups = ['0-6', '6-12', '12-18', '18-35', '35-65', '65-100']
min_prop_18_35 = {'A': 0.25, 'B': 0.3, 'C': 0.2}
Table_1_CityPopulation = {
    'A': 600, 'B': 650, 'C': 620, 'D': 600, 'E': 560,
    'F': 540, 'G': 590, 'H': 580, 'I': 610, 'J': 500
}
Table_2_PremiumData = [
    ['A', '0-6',   200,  900, 30, 70],
    ['A', '6-12',  200,  950, 28, 68],
    ['A', '12-18', 200, 1000, 28, 70],
    [None, None,   200, 1000, 35, 70],
    ['A', '65-100',200, 1100, 35, 75],
    ['B', '0-6',   180, 1150, 38, 80],
    ['B', '6-12',  180, 1180, 39, 75],
    ['B', '12-18', 180, 1200, 40, 75],
    ['B', '18-35', 180, 1200, 40, 80],
    ['B', '35-65', 180, 1200, 40, 85],
    [None, None,   220,  750, 25, 65],
    ['C', '6-12',  220,  780, 22, 63],
    ['C', '12-18', 220,  800, 20, 60],
    [None, '35-65',220,  850, 22, 63],
    ['C', '65-100',220,  875, 25, 65]
]

# Derived sets
cities = list(Table_1_CityPopulation.keys())  # ['A', 'B', ..., 'J']

# Build parameter dictionaries: premium, deductible, doc fee, specialist fee per (plan, age)
premium = {}
deductible = {}
doc_fee = {}
spec_fee = {}

# Initialize all (plan, age) combos as None (so we can detect missing values)
for p in policy_types:
    for a in age_groups:
        premium[(p, a)] = None
        deductible[(p, a)] = None
        doc_fee[(p, a)] = None
        spec_fee[(p, a)] = None

# Fill from Table_2_PremiumData when both plan and age are provided
for row in Table_2_PremiumData:
    plan, age, prem, ded, df, sf = row
    if plan is not None and age is not None:
        # Use exactly the values from the Parameters List (no modification)
        premium[(plan, age)] = prem
        deductible[(plan, age)] = ded
        doc_fee[(plan, age)] = df
        spec_fee[(plan, age)] = sf

# Minimum sub-quota per city: floor(P_i / 60)
import math
L = {}
for c in cities:
    P_i = Table_1_CityPopulation[c]
    L[c] = math.floor(P_i / 60)

# Total-period demand (sum of all city populations)
total_population = sum(Table_1_CityPopulation[c] for c in cities)

# =========================
# 2. Create Model
# =========================

model = gp.Model("Insurance_Plan_Allocation")

# =========================
# 3. Decision Variables
# =========================

# n[i,p,a] = number of customers in city i, plan p, age group a
n = model.addVars(
    cities, policy_types, age_groups,
    vtype=GRB.INTEGER,
    name="n"
)

# y[p] = 1 if plan p is used, 0 otherwise
y = model.addVars(policy_types, vtype=GRB.BINARY, name="y")

# =========================
# 4. Objective Function
# =========================

# Minimize:
# Z = sum_{i,p,a} (d_{p,a} + doc_{p,a} + spec_{p,a}) * n[i,p,a] + sum_p f_p * y[p]
cost_expr = gp.LinExpr()

for i in cities:
    for p in policy_types:
        for a in age_groups:
            if (deductible[(p, a)] is not None and
                doc_fee[(p, a)] is not None and
                spec_fee[(p, a)] is not None):
                per_cost = deductible[(p, a)] + doc_fee[(p, a)] + spec_fee[(p, a)]
                cost_expr += per_cost * n[i, p, a]
            else:
                # For missing (p,a) data, we will constrain n[i,p,a] = 0 later
                pass

# Add startup costs
for p in policy_types:
    cost_expr += startup_costs[p] * y[p]

model.setObjective(cost_expr, GRB.MINIMIZE)

# =========================
# 5. Constraints
# =========================

# 5.1 Demand satisfaction per city: sum_{p,a} n[i,p,a] = P_i
for i in cities:
    model.addConstr(
        gp.quicksum(n[i, p, a] for p in policy_types for a in age_groups)
        == Table_1_CityPopulation[i],
        name=f"demand_city_{i}"
    )

# 5.2 Total-period demand (redundant but included): sum_{i,p,a} n[i,p,a] = total_population
model.addConstr(
    gp.quicksum(n[i, p, a] for i in cities for p in policy_types for a in age_groups)
    == total_population,
    name="total_demand"
)

# 5.3 Average deductible limit:
# sum d_{p,a} * n[i,p,a] <= max_average_deductible * total_customers
left_deductible = gp.LinExpr()
for i in cities:
    for p in policy_types:
        for a in age_groups:
            if deductible[(p, a)] is not None:
                left_deductible += deductible[(p, a)] * n[i, p, a]

model.addConstr(
    left_deductible <= max_average_deductible *
    gp.quicksum(n[i, p, a] for i in cities for p in policy_types for a in age_groups),
    name="avg_deductible_limit"
)

# 5.4 Premium income requirement:
# sum premium_{p,a} * n[i,p,a] >= min_total_premium_income
premium_income = gp.LinExpr()
for i in cities:
    for p in policy_types:
        for a in age_groups:
            if premium[(p, a)] is not None:
                premium_income += premium[(p, a)] * n[i, p, a]

model.addConstr(
    premium_income >= min_total_premium_income,
    name="premium_income_requirement"
)

# 5.5 Age penetration constraints for 18-35:
# sum_i n[i,p,18-35] >= min_prop_18_35[p] * sum_{i,a} n[i,p,a]
age_18_35 = '18-35'
for p in policy_types:
    model.addConstr(
        gp.quicksum(n[i, p, age_18_35] for i in cities)
        >= min_prop_18_35[p] *
        gp.quicksum(n[i, p, a] for i in cities for a in age_groups),
        name=f"age_penetration_{p}"
    )

# 5.6 Minimum sub-quota per city-plan-age: n[i,p,a] >= floor(P_i/60)
for i in cities:
    for p in policy_types:
        for a in age_groups:
            model.addConstr(
                n[i, p, a] >= L[i],
                name=f"min_sub_quota_{i}_{p}_{a}"
            )

# 5.7 Plan activation linking using indicator constraints (no Big-M):
# If y[p] == 0, then sum_{i,a} n[i,p,a] == 0
for p in policy_types:
    total_plan_p = gp.quicksum(n[i, p, a] for i in cities for a in age_groups)
    model.addGenConstrIndicator(
        y[p], 0, total_plan_p == 0,
        name=f"plan_activation_{p}"
    )
    # For y[p]==1, no extra constraint is needed.

# 5.8 For missing (p,a) data, force n[i,p,a] = 0
for p in policy_types:
    for a in age_groups:
        if (premium[(p, a)] is None or
            deductible[(p, a)] is None or
            doc_fee[(p, a)] is None or
            spec_fee[(p, a)] is None):
            for i in cities:
                model.addConstr(
                    n[i, p, a] == 0,
                    name=f"forbid_missing_param_{i}_{p}_{a}"
                )

# =========================
# 6. Optimize
# =========================

model.Params.OutputFlag = 1
model.optimize()

# =========================
# 7. Print results and required final answer
# =========================

if model.Status == GRB.OPTIMAL:
    print(f"Optimal objective (total costs) = {model.ObjVal:.2f}")
    # Print plan usage
    for p in policy_types:
        print(f"Plan {p} used (y[{p}]) = {int(round(y[p].X))}")
    # Aggregate assignments per plan
    for p in policy_types:
        total_p = sum(n[i, p, a].X for i in cities for a in age_groups)
        print(f"Total customers in Plan {p}: {total_p:.0f}")
    final_answer = model.ObjVal  # total costs
else:
    print(f"No optimal solution found. Model status: {model.Status}")
    final_answer = float('nan')

# Required final output line
print(f"FinalAnswer=【{final_answer}】")