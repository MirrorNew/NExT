import gurobipy as gp
from gurobipy import GRB

# ==============================
# 1. Define parameters (from Parameters List)
# ==============================
delivery_quantities = [40, 60, 80]  # D_t
max_capacity = 100                  # C
cost_function_coeffs = {'a': 50, 'b': 0.002, 'exp': 2.9}
storage_cost = 4                    # h
initial_inventory = 0               # I_0

T = len(delivery_quantities)        # number of quarters, here 3

# ==============================
# 2. Create model
# ==============================
model = gp.Model("Engine_Production_Inventory_Planning")

# Allow general nonconvex power function x^2.9
model.Params.NonConvex = 2

# ==============================
# 3. Decision variables
# ==============================
# x_t: number of engines produced in quarter t (integer, 0 <= x_t <= max_capacity)
x = model.addVars(
    range(1, T + 1),
    vtype=GRB.INTEGER,
    lb=0,
    ub=max_capacity,
    name="x"
)

# I_t: inventory at end of quarter t (integer, I_t >= 0)
I = model.addVars(
    range(1, T + 1),
    vtype=GRB.INTEGER,
    lb=0,
    name="I"
)

# ==============================
# 4. Auxiliary substitution variables
# ==============================
# p_t: auxiliary variables for nonlinear term x_t^2.9
# Range: (-inf, +inf) as required
p = model.addVars(
    range(1, T + 1),
    vtype=GRB.CONTINUOUS,
    lb=-GRB.INFINITY,
    ub=GRB.INFINITY,
    name="p"
)

# Power constraints: p_t = x_t^2.9
for t in range(1, T + 1):
    model.addGenConstrPow(x[t], p[t], cost_function_coeffs['exp'])

# ==============================
# 5. Objective function
# ==============================
a = cost_function_coeffs['a']
b = cost_function_coeffs['b']

production_cost = gp.quicksum(a * x[t] + b * p[t] for t in range(1, T + 1))
inventory_cost = storage_cost * gp.quicksum(I[t] for t in range(1, T + 1))

model.setObjective(production_cost + inventory_cost, GRB.MINIMIZE)

# ==============================
# 6. Constraints
# ==============================

# Inventory balance constraints
# I_0 = initial_inventory (given as parameter)
# t = 1: I_1 = I_0 + x_1 - D_1 -> I_1 == x_1 - D_1  (since I_0 = 0)
model.addConstr(
    I[1] == initial_inventory + x[1] - delivery_quantities[0],
    name="InvBalance_1"
)

# For t >= 2: I_t = I_{t-1} + x_t - D_t
for t in range(2, T + 1):
    model.addConstr(
        I[t] == I[t - 1] + x[t] - delivery_quantities[t - 1],
        name=f"InvBalance_{t}"
    )

# Nonnegativity of inventory is already handled by lb=0 in variable definition
# Production bounds are already handled by lb and ub in variable definition

# ==============================
# 7. Solve the model and print results
# ==============================
model.optimize()

if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
    print("Optimal (or best found) solution:")
    total_cost = model.ObjVal
    for t in range(1, T + 1):
        print(f"Quarter {t}: produce x_{t} = {x[t].X:.0f}, ending inventory I_{t} = {I[t].X:.0f}")
    print(f"Total cost = {total_cost:.4f}")
else:
    print(f"Optimization ended with status {model.Status}")
    total_cost = float('nan')

# ==============================
# Final answer output statement
# Here, the "question answer" is interpreted as the minimum total cost.
# ==============================
print(f"FinalAnswer=【{total_cost}】")