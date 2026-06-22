import gurobipy as gp
from gurobipy import GRB

# ======================
# 1. Read Parameters List
# ======================

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

Table_1_monthly_demand = {
    'A': [20, 22, 25, 18, 15, 10],
    'B': [30, 28, 35, 32, 25, 20],
    'C': [15, 20, 30, 40, 20, 20],
    'D': [15, 20, 30, 20, 10, 10],
    'D_t': [80, 90, 120, 110, 70, 60]
}

Table_2_monthly_production_capacity = {
    'regular_capacity': {'A': 30, 'B': 30, 'C': 20, 'D': 20},
    'overtime_capacity': {'A': 6, 'B': 6, 'C': 4, 'D': 4},
    'regular_unit_price': {'A': 9, 'B': 11, 'C': 13, 'D': 10},
    'overtime_unit_price': {'A': 13.5, 'B': 16.5, 'C': 19.5, 'D': 15}
}

# Alias dictionaries strictly from Parameters List
demand = {i: Table_1_monthly_demand[i] for i in product_types}
RCap = Table_2_monthly_production_capacity['regular_capacity']
OCap = Table_2_monthly_production_capacity['overtime_capacity']
p_reg = Table_2_monthly_production_capacity['regular_unit_price']
p_ovt = Table_2_monthly_production_capacity['overtime_unit_price']

T = range(planning_horizon)  # 0..5 represent months 1..6

# As h_i and C_exp are not given in Parameters List, set them to 0 (neutral)
holding_cost = {i: 0.0 for i in product_types}
C_exp = 0.0

# ======================
# 2. Create Model
# ======================

model = gp.Model("Jinling_Production_Inventory_Planning")

# ======================
# 3. Decision Variables
# ======================

# Regular and overtime production
x_reg = model.addVars(product_types, T, vtype=GRB.INTEGER, lb=0, name="x_reg")
x_ovt = model.addVars(product_types, T, vtype=GRB.INTEGER, lb=0, name="x_ovt")

# Inventory
I = model.addVars(product_types, T, vtype=GRB.INTEGER, lb=0, name="I")

# Indicator: regular production at capacity (allows overtime)
b = model.addVars(product_types, T, vtype=GRB.BINARY, name="b")

# Free by-product B from A
f = model.addVars(T, vtype=GRB.INTEGER, lb=0, name="f")

# Excess B production above A pairing
eB = model.addVars(T, vtype=GRB.INTEGER, lb=0, name="eB")

# Indicator: combined C+D production exceeds 40 in month t
u = model.addVars(T, vtype=GRB.BINARY, name="u")

# ======================
# 4. Constraints
# ======================

# 4.1 Regular production capacity
for i in product_types:
    for t in T:
        model.addConstr(x_reg[i, t] <= RCap[i], name=f"RegCap_{i}_{t+1}")

# 4.2 Overtime capacity & usage condition (indicator constraints)
for i in product_types:
    for t in T:
        # If b[i,t] = 1 -> overtime allowed up to OCap[i]
        model.addGenConstrIndicator(
            b[i, t], 1, x_ovt[i, t] <= OCap[i],
            name=f"Ocap_on_{i}_{t+1}"
        )
        # If b[i,t] = 0 -> no overtime
        model.addGenConstrIndicator(
            b[i, t], 0, x_ovt[i, t] == 0,
            name=f"Ocap_off_{i}_{t+1}"
        )
        # If b[i,t] = 1 -> regular at capacity
        model.addGenConstrIndicator(
            b[i, t], 1, x_reg[i, t] >= RCap[i],
            name=f"RegFull_{i}_{t+1}"
        )
        # If b[i,t] = 0 -> regular <= capacity - 1
        model.addGenConstrIndicator(
            b[i, t], 0, x_reg[i, t] <= RCap[i] - 1,
            name=f"RegBelow_{i}_{t+1}"
        )

# 4.3 Free B by-product: f_t = 2 * x_reg[A,t]
for t in T:
    model.addConstr(
        f[t] == free_B_per_A * x_reg['A', t],
        name=f"FreeB_{t+1}"
    )

# 4.4 Inventory balance and nonnegativity (I >= 0 ensured by lb)
for i in product_types:
    for t in T:
        if t == 0:
            if i == 'B':
                model.addConstr(
                    I[i, t] == initial_inventory + x_reg[i, t] + x_ovt[i, t] + f[t] - demand[i][t],
                    name=f"Inv_{i}_{t+1}"
                )
            else:
                model.addConstr(
                    I[i, t] == initial_inventory + x_reg[i, t] + x_ovt[i, t] - demand[i][t],
                    name=f"Inv_{i}_{t+1}"
                )
        else:
            if i == 'B':
                model.addConstr(
                    I[i, t] == I[i, t-1] + x_reg[i, t] + x_ovt[i, t] + f[t] - demand[i][t],
                    name=f"Inv_{i}_{t+1}"
                )
            else:
                model.addConstr(
                    I[i, t] == I[i, t-1] + x_reg[i, t] + x_ovt[i, t] - demand[i][t],
                    name=f"Inv_{i}_{t+1}"
                )

# 4.5 Excess B production eB_t >= (xB(t) - xA(t))
for t in T:
    model.addConstr(
        eB[t] >= (x_reg['B', t] + x_ovt['B', t]) - (x_reg['A', t] + x_ovt['A', t]),
        name=f"ExcessB_{t+1}"
    )

# 4.6 Expansion trigger (C+D) using indicator constraints
for t in T:
    total_CD = x_reg['C', t] + x_ovt['C', t] + x_reg['D', t] + x_ovt['D', t]
    # If u[t] = 1 -> total_CD >= 41
    model.addGenConstrIndicator(
        u[t], 1, total_CD >= threshold_CD + 1,
        name=f"CD_exceed_{t+1}"
    )
    # If u[t] = 0 -> total_CD <= 40
    model.addGenConstrIndicator(
        u[t], 0, total_CD <= threshold_CD,
        name=f"CD_not_exceed_{t+1}"
    )

# 4.7 Expanded capacity C,D: x_reg[i,t] <= 20 + 16*u_{t-1}, t=2..6
for i in ['C', 'D']:
    # For t=1 (index 0): no previous u, capacity is 20
    model.addConstr(
        x_reg[i, 0] <= 20,
        name=f"ExpCap_{i}_1"
    )
    for t in range(1, planning_horizon):
        model.addConstr(
            x_reg[i, t] <= 20 + (expanded_capacity_CD - 20) * u[t-1],
            name=f"ExpCap_{i}_{t+1}"
        )

# ======================
# 5. Objective Function
# ======================

obj = gp.LinExpr()

for t in T:
    for i in product_types:
        if i in ['C', 'D']:
            # For C and D, unit prices change with expansion status (u_{t-1}) from next month
            if t == 0:
                # First month: no prior expansion
                eff_p_reg = p_reg[i]
                eff_p_ovt = p_ovt[i]
            else:
                # If u[t-1] = 1, use reduced prices; model via indicator constraints
                # We linearize cost directly using two linear terms:
                # cost = (normal_cost) + (reduction) * u[t-1]
                # regular:
                normal_reg = p_reg[i]
                reduced_reg = p_reg[i] - price_reduction_CD
                eff_p_reg = normal_reg + (reduced_reg - normal_reg) * u[t-1]
                # overtime:
                normal_ovt = p_ovt[i]
                reduced_ovt = overtime_price_multiplier * reduced_reg - overtime_price_reduction
                eff_p_ovt = normal_ovt + (reduced_ovt - normal_ovt) * u[t-1]
        else:
            eff_p_reg = p_reg[i]
            eff_p_ovt = p_ovt[i]

        obj += eff_p_reg * x_reg[i, t]
        obj += eff_p_ovt * x_ovt[i, t]

    # B excess surcharge (p_reg_B + 1)*eB_t
    obj += (p_reg['B'] + markup_unit_price_B) * eB[t]

    # Inventory holding costs
    for i in product_types:
        obj += holding_cost[i] * I[i, t]

    # Expansion cost C_exp * u_t
    obj += C_exp * u[t]

model.setObjective(obj, GRB.MINIMIZE)

# ======================
# 6. Solve Model
# ======================

model.Params.OutputFlag = 0
model.optimize()

# ======================
# 7. Print Results and Final Answer
# ======================

if model.status == GRB.OPTIMAL:
    total_cost = model.ObjVal

    # Optional detailed outputs
    print("Optimal total cost:", total_cost)
    print("\nRegular production:")
    for i in product_types:
        for t in T:
            val = x_reg[i, t].X
            if abs(val) > 1e-6:
                print(f"Month {t+1}, Product {i}, Regular: {val}")

    print("\nOvertime production:")
    for i in product_types:
        for t in T:
            val = x_ovt[i, t].X
            if abs(val) > 1e-6:
                print(f"Month {t+1}, Product {i}, Overtime: {val}")

    print("\nInventory:")
    for i in product_types:
        for t in T:
            val = I[i, t].X
            if abs(val) > 1e-6:
                print(f"Month {t+1}, Product {i}, Inventory: {val}")

    print("\nFree B by-product f[t]:")
    for t in T:
        val = f[t].X
        if abs(val) > 1e-6:
            print(f"Month {t+1}: {val}")

    print("\nExcess B production eB[t]:")
    for t in T:
        val = eB[t].X
        if abs(val) > 1e-6:
            print(f"Month {t+1}: {val}")

    print("\nC+D expansion trigger u[t]:")
    for t in T:
        print(f"Month {t+1}: u = {int(u[t].X)}")

    # Final required answer: total cost
    print(f"FinalAnswer=【{total_cost}】")
else:
    print("No optimal solution found.")
    print("FinalAnswer=【NaN】")