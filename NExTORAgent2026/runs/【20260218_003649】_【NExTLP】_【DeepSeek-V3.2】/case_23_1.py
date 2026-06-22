import gurobipy as gp

# ======================
# 1. PARAMETERS
# ======================
num_products = 4
product_types = ['A', 'B', 'C', 'D']
planning_horizon = 6
free_B_per_A = 2
markup_unit_price_B = 1
threshold_CD = 40
expanded_capacity_CD = 36
price_reduction_CD = 1
overtime_price_multiplier = 1.5
overtime_price_reduction = 1
initial_inventory = 0

# Demand data
demand = {
    'A': [20, 22, 25, 18, 15, 10],
    'B': [30, 28, 35, 32, 25, 20],
    'C': [15, 20, 30, 40, 20, 20],
    'D': [15, 20, 30, 20, 10, 10]
}

# Production capacity and prices
regular_capacity = {'A': 30, 'B': 30, 'C': 20, 'D': 20}
overtime_capacity = {'A': 6, 'B': 6, 'C': 4, 'D': 4}
regular_unit_price = {'A': 9, 'B': 11, 'C': 13, 'D': 10}
overtime_unit_price = {'A': 13.5, 'B': 16.5, 'C': 19.5, 'D': 15}

# Inventory holding costs (assumed, not provided in problem)
# Assuming 1 yuan per unit per month for all products
holding_cost = {'A': 1, 'B': 1, 'C': 1, 'D': 1}

# Big M for constraints
M = 1000

# ======================
# 2. CREATE MODEL
# ======================
model = gp.Model("Production_Inventory_Planning")

# ======================
# 3. DECISION VARIABLES
# ======================
# Regular production
x_regular = {}
for i in product_types:
    for t in range(planning_horizon):
        x_regular[i, t] = model.addVar(
            lb=0, ub=regular_capacity[i],
            vtype=gp.GRB.INTEGER,
            name=f"x_regular_{i}_{t}"
        )

# Overtime production
x_overtime = {}
for i in product_types:
    for t in range(planning_horizon):
        x_overtime[i, t] = model.addVar(
            lb=0, ub=overtime_capacity[i],
            vtype=gp.GRB.INTEGER,
            name=f"x_overtime_{i}_{t}"
        )

# Binary indicator for regular production at capacity
b = {}
for i in product_types:
    for t in range(planning_horizon):
        b[i, t] = model.addVar(
            vtype=gp.GRB.BINARY,
            name=f"b_{i}_{t}"
        )

# Ending inventory
I = {}
for i in product_types:
    for t in range(planning_horizon):
        I[i, t] = model.addVar(
            lb=0,
            vtype=gp.GRB.INTEGER,
            name=f"I_{i}_{t}"
        )

# Free by-product B units from A
f = {}
for t in range(planning_horizon):
    f[t] = model.addVar(
        lb=0,
        vtype=gp.GRB.INTEGER,
        name=f"f_{t}"
    )

# Excess B production above A pairing
e_B = {}
for t in range(planning_horizon):
    e_B[t] = model.addVar(
        lb=0,
        vtype=gp.GRB.INTEGER,
        name=f"e_B_{t}"
    )

# Binary indicator for C+D production exceeding threshold
u = {}
for t in range(planning_horizon):
    u[t] = model.addVar(
        vtype=gp.GRB.BINARY,
        name=f"u_{t}"
    )

# ======================
# 4. OBJECTIVE FUNCTION
# ======================
obj = gp.QuadExpr()

# Regular production cost
for i in product_types:
    for t in range(planning_horizon):
        obj += regular_unit_price[i] * x_regular[i, t]

# Overtime production cost
for i in product_types:
    for t in range(planning_horizon):
        obj += overtime_unit_price[i] * x_overtime[i, t]

# Additional cost for excess B production (regular price + 1)
for t in range(planning_horizon):
    obj += (regular_unit_price['B'] + markup_unit_price_B) * e_B[t]

# Inventory holding cost
for i in product_types:
    for t in range(planning_horizon):
        obj += holding_cost[i] * I[i, t]

# No explicit expansion cost term in objective (not provided in parameters)

model.setObjective(obj, gp.GRB.MINIMIZE)

# ======================
# 5. CONSTRAINTS
# ======================
# Constraint 1: Regular production capacity (already enforced by variable bounds)

# Constraint 2: Overtime production conditional on full regular capacity
for i in product_types:
    for t in range(planning_horizon):
        # If b_{i,t} = 1, then x_regular_{i,t} >= regular_capacity[i]
        model.addGenConstrIndicator(
            b[i, t], 1,
            x_regular[i, t] >= regular_capacity[i],
            name=f"indicator_regular_at_capacity_{i}_{t}"
        )
        # If b_{i,t} = 0, then x_regular_{i,t} <= regular_capacity[i] - 1
        model.addGenConstrIndicator(
            b[i, t], 0,
            x_regular[i, t] <= regular_capacity[i] - 1,
            name=f"indicator_regular_not_at_capacity_{i}_{t}"
        )
        # Overtime production only if regular at capacity
        model.addConstr(
            x_overtime[i, t] <= overtime_capacity[i] * b[i, t],
            name=f"overtime_conditional_{i}_{t}"
        )

# Constraint 3: Inventory balance
for i in product_types:
    for t in range(planning_horizon):
        if t == 0:
            inventory_previous = initial_inventory
        else:
            inventory_previous = I[i, t-1]
        
        # For product B, add free by-product
        if i == 'B':
            model.addConstr(
                I[i, t] == inventory_previous + x_regular[i, t] + x_overtime[i, t] + f[t] - demand[i][t],
                name=f"inventory_balance_{i}_{t}"
            )
        else:
            model.addConstr(
                I[i, t] == inventory_previous + x_regular[i, t] + x_overtime[i, t] - demand[i][t],
                name=f"inventory_balance_{i}_{t}"
            )

# Constraint 4: Free B by-product from A regular production
for t in range(planning_horizon):
    model.addConstr(
        f[t] == free_B_per_A * x_regular['A', t],
        name=f"free_B_byproduct_{t}"
    )

# Constraint 5: Excess B production definition
for t in range(planning_horizon):
    total_B = x_regular['B', t] + x_overtime['B', t]
    total_A = x_regular['A', t] + x_overtime['A', t]
    model.addConstr(
        e_B[t] >= total_B - total_A,
        name=f"excess_B_definition_{t}"
    )

# Constraint 6: Expansion trigger for C+D production
for t in range(planning_horizon):
    total_CD = (x_regular['C', t] + x_overtime['C', t] + 
                x_regular['D', t] + x_overtime['D', t])
    
    # If total_CD > 40, then u_t = 1
    model.addConstr(
        total_CD <= threshold_CD + M * u[t],
        name=f"CD_trigger_upper_{t}"
    )
    model.addConstr(
        total_CD >= (threshold_CD + 1) * u[t],
        name=f"CD_trigger_lower_{t}"
    )

# Constraint 7: Expanded regular capacity for C and D (from month 2 onward)
for t in range(1, planning_horizon):
    # For product C
    model.addConstr(
        x_regular['C', t] <= regular_capacity['C'] + (expanded_capacity_CD - regular_capacity['C']) * u[t-1],
        name=f"expanded_capacity_C_{t}"
    )
    # For product D
    model.addConstr(
        x_regular['D', t] <= regular_capacity['D'] + (expanded_capacity_CD - regular_capacity['D']) * u[t-1],
        name=f"expanded_capacity_D_{t}"
    )

# ======================
# 6. SOLVE
# ======================
model.optimize()

# ======================
# 7. RESULTS
# ======================
if model.status == gp.GRB.OPTIMAL:
    total_cost = model.objVal
    print("Optimal solution found.")
    print(f"Total cost: {total_cost}")
    
    # Print production plan summary
    print("\n=== Production Plan ===")
    for t in range(planning_horizon):
        print(f"Month {t+1}:")
        for i in product_types:
            reg = x_regular[i, t].X
            ovt = x_overtime[i, t].X
            if reg > 0 or ovt > 0:
                print(f"  {i}: Regular={reg}, Overtime={ovt}")
    
    print("\n=== Inventory ===")
    for i in product_types:
        inv_str = ", ".join([f"{I[i, t].X:.0f}" for t in range(planning_horizon)])
        print(f"  {i}: [{inv_str}]")
    
    print("\n=== Other Variables ===")
    for t in range(planning_horizon):
        print(f"Month {t+1}: f_t={f[t].X}, e_B_t={e_B[t].X}, u_t={u[t].X}")
    
    print(f"\nFinalAnswer=【{total_cost}】")
else:
    print("No optimal solution found.")
    print(f"FinalAnswer=【0】")