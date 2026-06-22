import gurobipy as gp
from gurobipy import GRB

# ==========================
# 1. Define Parameters
# ==========================

# Sets
T = ['January', 'February', 'March', 'April', 'May', 'June']  # months
J = 7  # number of products
products = range(J)  # 0..6 for Python indexing

# Machine counts
num_grinders = 4
num_vertical_drills = 2
num_horizontal_drills = 3
num_boring_machines = 1
num_planers = 1

# Repair schedule (as provided)
repair_schedule = {
    'January':   {'grinder': 1},
    'February':  {'horizontal drill': 2},
    'March':     {'boring machine': 1},
    'April':     {'vertical drill': 1},
    'May':       {'grinder': 1, 'vertical drill': 1},
    'June':      {'planer': 1, 'horizontal drill': 1}
}

storage_fee_per_unit = 5
storage_capacity_per_product = 100
initial_inventory = [0, 0, 0, 0, 0, 0, 0]
final_inventory_requirement = [50, 50, 50, 50, 50, 50, 50]
days_per_month = 24
shifts_per_day = 2
hours_per_shift = 8
storage_fee_end_of_June = 0  # not used explicitly but kept for completeness

# Table C-3: processing times and profits
Table_1_C3 = {
    'Grinding machine': [0.5, 0.7, None, None, 0.3, 0.2, 0.5],
    'Vertical drill':   [0.1, 0.2, None, 0.3, None, 0.6, None],
    'Horizontal drill': [0.2, None, 0.8, None, None, None, 0.6],
    'Boring machine':   [0.05, 0.03, None, 0.07, 0.1, None, 0.08],
    'Planer':           [None, None, 0.01, None, 0.05, None, 0.05],
    'Profit per piece': [100, 60, 80, 40, 110, 90, 30]
}

# Demand table C-4
Table_2_C4 = {
    'January':  [500, 1000, 300, 300, 800, 200, 100],
    'February': [600, 500, 200, 0,   400, 300, 150],
    'March':    [300, 600, 0,   0,   500, 400, 100],
    'April':    [200, 300, 400, 500, 200, 0,   100],
    'May':      [0,   100, 500, 100, 1000,300, 0],
    'June':     [500, 500, 100, 300, 1100,500, 60]
}

# Derived parameters

# Hours per machine per month
hours_per_machine_per_month = days_per_month * shifts_per_day * hours_per_shift  # 384

# Processing time matrices (replace None with 0.0)
g_time = [t if t is not None else 0.0 for t in Table_1_C3['Grinding machine']]
v_time = [t if t is not None else 0.0 for t in Table_1_C3['Vertical drill']]
h_time = [t if t is not None else 0.0 for t in Table_1_C3['Horizontal drill']]
b_time = [t if t is not None else 0.0 for t in Table_1_C3['Boring machine']]
p_time = [t if t is not None else 0.0 for t in Table_1_C3['Planer']]

# Profit per unit
profit = Table_1_C3['Profit per piece']

# Demand d[t][j]
demand = {t: Table_2_C4[t] for t in T}

# Machine repair counts per month
rG = {t: 0 for t in T}
rV = {t: 0 for t in T}
rH = {t: 0 for t in T}
rB = {t: 0 for t in T}
rP = {t: 0 for t in T}

for month in T:
    if 'grinder' in repair_schedule.get(month, {}):
        rG[month] = repair_schedule[month]['grinder']
    if 'vertical drill' in repair_schedule.get(month, {}):
        rV[month] = repair_schedule[month]['vertical drill']
    if 'horizontal drill' in repair_schedule.get(month, {}):
        rH[month] = repair_schedule[month]['horizontal drill']
    if 'boring machine' in repair_schedule.get(month, {}):
        rB[month] = repair_schedule[month]['boring machine']
    if 'planer' in repair_schedule.get(month, {}):
        rP[month] = repair_schedule[month]['planer']

# ==========================
# 2. Create Model
# ==========================

model = gp.Model("Sunshine_Machinery_Production_Planning")

# ==========================
# 3. Decision Variables
# ==========================

# x[t,j]: production quantity
x = model.addVars(
    T, products,
    vtype=GRB.CONTINUOUS,
    lb=0.0,
    name="x"
)

# s[t,j]: sales quantity
s = model.addVars(
    T, products,
    vtype=GRB.CONTINUOUS,
    lb=0.0,
    name="s"
)

# I[t,j]: inventory at end of month t
I = model.addVars(
    T, products,
    vtype=GRB.CONTINUOUS,
    lb=0.0,
    name="I"
)

# For initial inventory I_0,j, we will treat them explicitly in constraints (constants)

# ==========================
# 4. Objective Function
# ==========================

# Maximize sum_t sum_j profit[j]*s[t,j] - 5*sum_t<=May sum_j I[t,j]
revenue_expr = gp.quicksum(profit[j] * s[t, j] for t in T for j in products)

# Storage cost: only for January to May (T[0:5])
storage_cost_expr = gp.quicksum(
    storage_fee_per_unit * I[t, j]
    for t in T[:-1]  # all except June
    for j in products
)

model.setObjective(revenue_expr - storage_cost_expr, GRB.MAXIMIZE)

# ==========================
# 5. Constraints
# ==========================

# Helper for initial inventory (month index -1 equivalent)
I0 = {j: initial_inventory[j] for j in products}

# 5.1 Machine capacity constraints

for t in T:
    # Grinder capacity
    model.addConstr(
        gp.quicksum(g_time[j] * x[t, j] for j in products)
        <= (num_grinders - rG[t]) * hours_per_machine_per_month,
        name=f"Grinding_Capacity_{t}"
    )

    # Vertical drill capacity
    model.addConstr(
        gp.quicksum(v_time[j] * x[t, j] for j in products)
        <= (num_vertical_drills - rV[t]) * hours_per_machine_per_month,
        name=f"Vertical_Drill_Capacity_{t}"
    )

    # Horizontal drill capacity
    model.addConstr(
        gp.quicksum(h_time[j] * x[t, j] for j in products)
        <= (num_horizontal_drills - rH[t]) * hours_per_machine_per_month,
        name=f"Horizontal_Drill_Capacity_{t}"
    )

    # Boring machine capacity
    model.addConstr(
        gp.quicksum(b_time[j] * x[t, j] for j in products)
        <= (num_boring_machines - rB[t]) * hours_per_machine_per_month,
        name=f"Boring_Machine_Capacity_{t}"
    )

    # Planer capacity
    model.addConstr(
        gp.quicksum(p_time[j] * x[t, j] for j in products)
        <= (num_planers - rP[t]) * hours_per_machine_per_month,
        name=f"Planer_Capacity_{t}"
    )

# 5.2 Demand limit and sales vs availability + inventory balance

# We need to manage I_{t-1,j}; define an ordered list of months
for idx, t in enumerate(T):
    for j in products:
        # Demand limit
        model.addConstr(
            s[t, j] <= demand[t][j],
            name=f"Demand_Limit_{t}_{j+1}"
        )

        # Inventory balance and availability
        if idx == 0:
            # Month = January: use initial inventory I0
            # Sales limited by production plus initial inventory
            model.addConstr(
                s[t, j] <= x[t, j] + I0[j],
                name=f"Supply_vs_Sales_{t}_{j+1}"
            )
            # Inventory balance: I[January,j] = I0[j] + x - s
            model.addConstr(
                I[t, j] == I0[j] + x[t, j] - s[t, j],
                name=f"Inventory_Balance_{t}_{j+1}"
            )
        else:
            # For months February..June
            prev_t = T[idx - 1]
            # Sales limited by production plus inventory from previous month
            model.addConstr(
                s[t, j] <= x[t, j] + I[prev_t, j],
                name=f"Supply_vs_Sales_{t}_{j+1}"
            )
            # Inventory balance: I[t,j] = I[t-1,j] + x[t,j] - s[t,j]
            model.addConstr(
                I[t, j] == I[prev_t, j] + x[t, j] - s[t, j],
                name=f"Inventory_Balance_{t}_{j+1}"
            )

# 5.3 Inventory capacity
for t in T:
    for j in products:
        model.addConstr(
            I[t, j] <= storage_capacity_per_product,
            name=f"Inventory_Capacity_{t}_{j+1}"
        )

# 5.4 Final inventory requirement at end of June
last_month = 'June'
for j in products:
    model.addConstr(
        I[last_month, j] == final_inventory_requirement[j],
        name=f"Final_Inventory_Requirement_{last_month}_{j+1}"
    )

# ==========================
# 6. Solve Model
# ==========================

model.optimize()

# ==========================
# 7. Print Results
# ==========================

if model.status == GRB.OPTIMAL:
    print(f"Optimal objective (maximum total profit) = {model.objVal:.2f}")
    # (Optional detailed outputs) Uncomment if needed:
    # for t in T:
    #     print(f"Month: {t}")
    #     for j in products:
    #         print(
    #             f"  Product {j+1}: "
    #             f"Produce {x[t, j].X:.2f}, "
    #             f"Sell {s[t, j].X:.2f}, "
    #             f"Inventory {I[t, j].X:.2f}"
    #         )
    total_profit = model.objVal
else:
    total_profit = float('nan')
    print("No optimal solution found.")

# Required final answer print (maximum total profit)
print(f"FinalAnswer=【{total_profit}】")