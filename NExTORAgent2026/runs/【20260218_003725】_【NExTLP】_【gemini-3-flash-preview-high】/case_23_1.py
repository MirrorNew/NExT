import gurobipy as gp
from gurobipy import GRB

# 1. Initialize the model
model = gp.Model("JinlingFactoryProduction")

# 2. Parameters from the problem description
num_products = 4
product_types = ['A', 'B', 'C', 'D']
planning_horizon = 6
free_B_per_A = 2
markup_B = 1.0  # markup_unit_price_B
threshold_CD = 40
expanded_cap_CD = 36
price_reduction_CD = 1.0
ot_multiplier = 1.5

demand = {
    'A': [20, 22, 25, 18, 15, 10],
    'B': [30, 28, 35, 32, 25, 20],
    'C': [15, 20, 30, 40, 20, 20],
    'D': [15, 20, 30, 20, 10, 10]
}

# Base regular capacity
reg_cap_base = {'A': 30, 'B': 30, 'C': 20, 'D': 20}
# Overtime capacity limit
ot_cap_base = {'A': 6, 'B': 6, 'C': 4, 'D': 4}
# Regular unit price
reg_price_base = {'A': 9, 'B': 11, 'C': 13, 'D': 10}
# Overtime unit price (base)
ot_price_base = {'A': 13.5, 'B': 16.5, 'C': 19.5, 'D': 15}

# 3. Decision Variables
xr = model.addVars(product_types, range(planning_horizon), vtype=GRB.INTEGER, name="xr")
xo = model.addVars(product_types, range(planning_horizon), vtype=GRB.INTEGER, name="xo")
inv = model.addVars(product_types, range(planning_horizon), vtype=GRB.INTEGER, name="inv")
b = model.addVars(product_types, range(planning_horizon), vtype=GRB.BINARY, name="b")
u = model.addVars(range(planning_horizon), vtype=GRB.BINARY, name="u")
s = model.addVars(range(-1, planning_horizon), vtype=GRB.BINARY, name="s")
eb = model.addVars(range(planning_horizon), vtype=GRB.INTEGER, name="eb")

# Variables to store costs (needed for indicator constraints)
cost_reg = model.addVars(product_types, range(planning_horizon), vtype=GRB.CONTINUOUS, name="cost_reg")
cost_ot = model.addVars(product_types, range(planning_horizon), vtype=GRB.CONTINUOUS, name="cost_ot")

# Helper variables for logical conditions
zero_var = model.addVar(vtype=GRB.INTEGER, lb=0, ub=0, name="zero_var")

# 4. Constraints
# Initial state of expansion
model.addConstr(s[-1] == 0)

for t in range(planning_horizon):
    # Overtime production condition: Regular production must reach capacity limit
    # Product A and B
    for i in ['A', 'B']:
        model.addConstr(xr[i, t] <= reg_cap_base[i])
        model.addConstr(xr[i, t] >= reg_cap_base[i] * b[i, t])
        model.addConstr(xo[i, t] <= ot_cap_base[i] * b[i, t])
        model.addConstr(cost_reg[i, t] == reg_price_base[i] * xr[i, t])
        model.addConstr(cost_ot[i, t] == ot_price_base[i] * xo[i, t])

    # Product C and D
    for i in ['C', 'D']:
        # Capacity limit based on expansion state in previous month
        model.addGenConstrIndicator(s[t-1], 0, xr[i, t] <= 20)
        model.addGenConstrIndicator(s[t-1], 0, xr[i, t] >= 20 * b[i, t])
        model.addGenConstrIndicator(s[t-1], 1, xr[i, t] <= expanded_cap_CD)
        model.addGenConstrIndicator(s[t-1], 1, xr[i, t] >= expanded_cap_CD * b[i, t])
        model.addConstr(xo[i, t] <= ot_cap_base[i] * b[i, t])
        
        # Unit prices and cost calculation based on expansion state
        # Price reduction: all unit prices reduced by 1
        model.addGenConstrIndicator(s[t-1], 0, cost_reg[i, t] == reg_price_base[i] * xr[i, t])
        model.addGenConstrIndicator(s[t-1], 0, cost_ot[i, t] == ot_price_base[i] * xo[i, t])
        model.addGenConstrIndicator(s[t-1], 1, cost_reg[i, t] == (reg_price_base[i] - 1) * xr[i, t])
        model.addGenConstrIndicator(s[t-1], 1, cost_ot[i, t] == (reg_price_base[i] * 1.5 - 1) * xo[i, t])

    # Expansion Trigger: If sum of C and D production exceeds 40
    cd_prod_sum = model.addVar(vtype=GRB.INTEGER)
    model.addConstr(cd_prod_sum == xr['C', t] + xo['C', t] + xr['D', t] + xo['D', t])
    model.addGenConstrIndicator(u[t], 1, cd_prod_sum >= threshold_CD + 1)
    model.addGenConstrIndicator(u[t], 0, cd_prod_sum <= threshold_CD)

    # State update: Expansion is permanent once triggered
    model.addConstr(s[t] >= s[t-1])
    model.addConstr(s[t] >= u[t])
    model.addConstr(s[t] <= s[t-1] + u[t])

    # Free product B calculation
    f_t = free_B_per_A * xr['A', t]

    # Inventory Balance and Non-negativity
    for i in product_types:
        prev_inv = inv[i, t-1] if t > 0 else 0
        if i == 'B':
            model.addConstr(inv[i, t] == prev_inv + xr[i, t] + xo[i, t] + f_t - demand[i][t])
        else:
            model.addConstr(inv[i, t] == prev_inv + xr[i, t] + xo[i, t] - demand[i][t])
        model.addConstr(inv[i, t] >= 0)

    # Excess B production calculation (Paid B vs Paid A)
    diff_b_a = model.addVar(lb=-GRB.INFINITY, vtype=GRB.INTEGER)
    model.addConstr(diff_b_a == (xr['B', t] + xo['B', t]) - (xr['A', t] + xo['A', t]))
    model.addGenConstrMax(eb[t], [diff_b_a, zero_var])

# 5. Set Up Objective Function
# Objective = Σ (Regular costs + Overtime costs) + Σ (markup_B + reg_price_B) * eb
# No inventory holding costs or expansion costs provided in the parameter list values.
total_cost = gp.quicksum(cost_reg[i, t] + cost_ot[i, t] for i in product_types for t in range(planning_horizon)) + \
             gp.quicksum((reg_price_base['B'] + markup_B) * eb[t] for t in range(planning_horizon))

model.setObjective(total_cost, GRB.MINIMIZE)

# 6. Solve and Print Result
model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"FinalAnswer=【{model.ObjVal}】")