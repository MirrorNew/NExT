import gurobipy as gp
from gurobipy import GRB

# 1. Define Parameters from the provided Parameters List
num_products = 4
product_types = ['A', 'B', 'C', 'D']
planning_horizon = 6
free_B_per_A = 2
markup_unit_price_B = 1.0
threshold_CD = 40
expanded_capacity_CD = 36
price_reduction_CD = 1.0
overtime_price_multiplier = 1.5
overtime_price_reduction = 1.0
initial_inventory = 0

# Tables
monthly_demand = {'A': [20, 22, 25, 18, 15, 10], 
                  'B': [30, 28, 35, 32, 25, 20], 
                  'C': [15, 20, 30, 40, 20, 20], 
                  'D': [15, 20, 30, 20, 10, 10]}

regular_capacity = {'A': 30, 'B': 30, 'C': 20, 'D': 20}
overtime_capacity = {'A': 6, 'B': 6, 'C': 4, 'D': 4}
regular_unit_price = {'A': 9, 'B': 11, 'C': 13, 'D': 10}
overtime_unit_price = {'A': 13.5, 'B': 16.5, 'C': 19.5, 'D': 15}

# Since holding cost is not explicitly provided in the parameters list, 
# and the objective mentions minimizing it, we assume it is 0 based on strict adherence 
# to the provided values. If there were values, they would be used here.
holding_cost = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

# Months 1 to 6
months = range(planning_horizon)
products = product_types

# 2. Create Model
model = gp.Model("ProductionPlanning")

# 3. Create Decision Variables

# Production variables
x_r = model.addVars(products, months, vtype=GRB.INTEGER, lb=0, name="RegProd")
x_o = model.addVars(products, months, vtype=GRB.INTEGER, lb=0, name="OverProd")

# Inventory variables
I = model.addVars(products, months, vtype=GRB.INTEGER, lb=0, name="Inventory")

# Indicator for overtime usage (1 if regular capacity full)
b = model.addVars(products, months, vtype=GRB.BINARY, name="OvertimeIndicator")

# Expansion logic variables
# u[t]: 1 if C+D production > 40 in month t
u = model.addVars(months, vtype=GRB.BINARY, name="ExpansionTrigger")
# z[t]: 1 if expansion is active in month t (result of previous triggers)
z = model.addVars(months, vtype=GRB.BINARY, name="ExpansionState")

# Excess B variable
e_B = model.addVars(months, vtype=GRB.INTEGER, lb=0, name="ExcessB")

# Discount variable for C and D (linearization of z[t] * production)
# discount[i, t] represents the amount of production eligible for price reduction
d_CD = model.addVars(['C', 'D'], months, vtype=GRB.CONTINUOUS, lb=0, name="DiscountQty")

# Big M for constraints
M = 1000

# 4. Set Objective Function
# Minimize: Regular Cost + Overtime Cost + Excess B Penalty - Price Reductions + Inventory Cost
# Note: "Expansion Cost" term in objective text is handled via dynamic costs; no fixed cost param provided.

total_cost = 0
for t in months:
    # Base Production Costs
    for i in products:
        total_cost += regular_unit_price[i] * x_r[i, t]
        total_cost += overtime_unit_price[i] * x_o[i, t]
        total_cost += holding_cost[i] * I[i, t]
    
    # Excess B Penalty
    total_cost += markup_unit_price_B * e_B[t]
    
    # Price Reductions for C and D if expanded
    # Reduction is 1 per unit for Regular and 1 per unit for Overtime (based on params)
    for i in ['C', 'D']:
        total_cost -= price_reduction_CD * d_CD[i, t] # Regular reduction
        # For overtime, param says reduction is 1. Since d_CD captures total prod, we apply 1.
        # However, we need to be careful. d_CD should capture (x_r + x_o) multiplied by z[t].
        # The reduction is 1 per unit regardless of reg or ot.

model.setObjective(total_cost, GRB.MINIMIZE)

# 5. Constraints

# Initial inventory assumption: I[i, -1] = 0
prev_inv = {i: initial_inventory for i in products}

for t in months:
    # Inventory Balance
    for i in products:
        inflow = x_r[i, t] + x_o[i, t]
        if i == 'B':
            # Add free B from A's regular production
            inflow += free_B_per_A * x_r['A', t]
        
        if t == 0:
            model.addConstr(I[i, t] == prev_inv[i] + inflow - monthly_demand[i][t], f"Bal_{i}_{t}")
        else:
            model.addConstr(I[i, t] == I[i, t-1] + inflow - monthly_demand[i][t], f"Bal_{i}_{t}")

    # Capacity Constraints (Regular)
    # A and B are static
    model.addConstr(x_r['A', t] <= regular_capacity['A'], f"CapReg_A_{t}")
    model.addConstr(x_r['B', t] <= regular_capacity['B'], f"CapReg_B_{t}")
    
    # C and D are dynamic based on z[t]
    # Cap = 20 + 16 * z[t] -> If z=0 cap=20, if z=1 cap=36
    for i in ['C', 'D']:
        model.addConstr(x_r[i, t] <= regular_capacity[i] + (expanded_capacity_CD - regular_capacity[i]) * z[t], f"CapReg_{i}_{t}")

    # Overtime Capacity and Usage Condition
    for i in products:
        # Overtime limit
        model.addConstr(x_o[i, t] <= overtime_capacity[i] * b[i, t], f"CapOver_{i}_{t}")
        
        # If overtime is used (b=1), regular must be full
        # Use indicator constraints
        if i in ['A', 'B']:
            current_cap = regular_capacity[i]
            model.addGenConstrIndicator(b[i, t], 1, x_r[i, t] >= current_cap, name=f"ForceFullReg_{i}_{t}")
        else:
            # For C/D, capacity depends on z[t]
            # x_r >= 20 + 16 * z[t]  <=>  x_r - 16 * z[t] >= 20
            model.addGenConstrIndicator(b[i, t], 1, x_r[i, t] - (expanded_capacity_CD - regular_capacity[i]) * z[t] >= regular_capacity[i], name=f"ForceFullReg_{i}_{t}")

    # Excess B Calculation
    # e_B >= (Prod_B - Prod_A)
    prod_A = x_r['A', t] + x_o['A', t]
    prod_B = x_r['B', t] + x_o['B', t]
    model.addConstr(e_B[t] >= prod_B - prod_A, f"ExcessB_{t}")

    # Expansion Logic (C and D)
    sum_CD_prod = x_r['C', t] + x_o['C', t] + x_r['D', t] + x_o['D', t]
    
    # u[t] = 1 <-> sum > 40
    # sum <= 40 + M * u[t]  (if u=0, sum<=40)
    model.addConstr(sum_CD_prod <= threshold_CD + M * u[t], f"TriggerUB_{t}")
    # sum >= 41 - M * (1-u[t]) (if u=1, sum>=41)
    model.addConstr(sum_CD_prod >= (threshold_CD + 1) - M * (1 - u[t]), f"TriggerLB_{t}")

    # z[t] Logic
    if t == 0:
        model.addConstr(z[t] == 0, "InitState_0")
    else:
        # z[t] = z[t-1] OR u[t-1]
        # z[t] >= z[t-1]
        model.addConstr(z[t] >= z[t-1], f"StatePersist_{t}")
        # z[t] >= u[t-1]
        model.addConstr(z[t] >= u[t-1], f"StateTrigger_{t}")
        # z[t] <= z[t-1] + u[t-1] (To prevent arbitrary activation)
        model.addConstr(z[t] <= z[t-1] + u[t-1], f"StateBound_{t}")

    # Discount Linearization for C and D
    # d_CD[i,t] = z[t] * (x_r + x_o)
    # We are minimizing Cost - Discount => Maximizing Discount.
    # So we need to bound d_CD from above.
    for i in ['C', 'D']:
        total_prod_i = x_r[i, t] + x_o[i, t]
        model.addConstr(d_CD[i, t] <= total_prod_i, f"Disc_UB1_{i}_{t}")
        model.addConstr(d_CD[i, t] <= M * z[t], f"Disc_UB2_{i}_{t}")
        # Since we maximize d_CD (via min -d_CD), the solver will push it to min(prod, M*z).
        # If z=1, d=prod. If z=0, d=0. Correct.

# 6. Solve
model.optimize()

# 7. Print Results
if model.status == GRB.OPTIMAL:
    obj_val = model.objVal
    print(f"FinalAnswer=【{obj_val}】")
else:
    print("No optimal solution found.")