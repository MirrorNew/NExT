import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters and Data
# =========================

number_of_factories = 5
total_orders = 12
standard_orders = 9
joint_orders = 3
factories_per_joint_order = 2
min_orders_per_factory = 1
max_orders_per_factory = 4
full_fulfillment_required = True
bonus_threshold = 3
bonus_rate = 0.1
bonus_base = 'lowest_cost_assigned_order'

Table_1_Cost = [
    [9,   2,  7,  8,  None, 6, 5, 4, 3, 11.2, 9.2, 10],
    [6,   4,  3,  None, None, 5, 7, 8, 9, 11.3, 11.4, 11.1],
    [5,   8,  1,  8,  None, 4, 6, 7, 2, 12.0, 10.4, 12.4],
    [7,   6,  9,  4,  3,    2, 5, 8, 7, 9.9,  10.8, 12.2],
    [8,   5,  6,  4,  9,    7, 3, 2, 1, 11.4, 10.6, 13.1]
]

# Sets (0-based indices for Python, but conceptually factories 1..5, orders 1..12)
factories = range(number_of_factories)
orders = range(total_orders)
standard_order_indices = range(standard_orders)  # 0..8 for orders 1..9
joint_order_indices = range(standard_orders, total_orders)  # 9..11 for orders 10..12

# Build a cost matrix and qualification flags
# Use a large constant M for cost-related constraints; take max finite cost
finite_costs = [Table_1_Cost[i][j]
                for i in factories for j in orders
                if Table_1_Cost[i][j] is not None]
M = max(finite_costs)

# =========================
# 2. Create Model
# =========================

model = gp.Model("Haitong_Order_Factory_Assignment")

# =========================
# 3. Decision Variables
# =========================

# x[i,j] = 1 if factory i is assigned to order j
x = model.addVars(
    factories,
    orders,
    vtype=GRB.BINARY,
    name="x"
)

# y[i] = 1 if factory i undertakes >= bonus_threshold (3) orders
y = model.addVars(
    factories,
    vtype=GRB.BINARY,
    name="y"
)

# m[i] = minimum cost among orders assigned to factory i (only relevant if y[i]=1)
m = model.addVars(
    factories,
    vtype=GRB.CONTINUOUS,
    lb=0.0,
    name="m"
)

# =========================
# 4. Objective Function
# =========================

# Total assignment cost (only for qualified pairs)
assignment_cost = gp.quicksum(
    Table_1_Cost[i][j] * x[i, j]
    for i in factories for j in orders
    if Table_1_Cost[i][j] is not None
)

# Performance bonus: bonus_rate * sum_i y_i * m_i
bonus = bonus_rate * gp.quicksum(y[i] * m[i] for i in factories)

model.setObjective(assignment_cost - bonus, GRB.MINIMIZE)

# =========================
# 5. Constraints
# =========================

# 5.1 Standard-order assignment: each standard order served by exactly one factory
if full_fulfillment_required:
    for j in standard_order_indices:
        model.addConstr(
            gp.quicksum(x[i, j] for i in factories
                        if Table_1_Cost[i][j] is not None) == 1,
            name=f"StdAssign_order{j+1}"
        )

# 5.2 Joint-order assignment: each joint order served by exactly 'factories_per_joint_order' factories
if full_fulfillment_required:
    for j in joint_order_indices:
        model.addConstr(
            gp.quicksum(x[i, j] for i in factories
                        if Table_1_Cost[i][j] is not None) == factories_per_joint_order,
            name=f"JointAssign_order{j+1}"
        )

# 5.3 Min orders per factory
for i in factories:
    model.addConstr(
        gp.quicksum(
            x[i, j] for j in orders if Table_1_Cost[i][j] is not None
        ) >= min_orders_per_factory,
        name=f"MinOrders_fac{i+1}"
    )

# 5.4 Max orders per factory
for i in factories:
    model.addConstr(
        gp.quicksum(
            x[i, j] for j in orders if Table_1_Cost[i][j] is not None
        ) <= max_orders_per_factory,
        name=f"MaxOrders_fac{i+1}"
    )

# 5.5 Qualification: x[i,j] = 0 if unqualified (cost is None)
for i in factories:
    for j in orders:
        if Table_1_Cost[i][j] is None:
            model.addConstr(x[i, j] == 0, name=f"Qual_fac{i+1}_ord{j+1}")

# 5.6 Link y[i] with number of orders assigned to factory i
#     sum_j x_ij >= bonus_threshold * y_i
#     sum_j x_ij <= 2 + 2*y_i     (given model)
for i in factories:
    sum_x_i = gp.quicksum(
        x[i, j] for j in orders if Table_1_Cost[i][j] is not None
    )
    model.addConstr(sum_x_i >= bonus_threshold * y[i],
                    name=f"YLower_fac{i+1}")
    model.addConstr(sum_x_i <= 2 + 2 * y[i],
                    name=f"YUpper_fac{i+1}")

# 5.7 Min-cost capturing using indicator constraints

# 5.7.1 m_i <= c_ij if x_ij = 1; if x_ij = 0, m_i can be up to M
# Use addGenConstrIndicator instead of big-M linear form
for i in factories:
    for j in orders:
        if Table_1_Cost[i][j] is not None:
            cij = Table_1_Cost[i][j]
            # Indicator: if x[i,j] == 1 then m[i] <= c_ij
            model.addGenConstrIndicator(
                x[i, j], 1,
                m[i] <= cij,
                name=f"MinCostActive_fac{i+1}_ord{j+1}"
            )
            # When x[i,j] == 0, we do not impose a tighter bound than M via indicator;
            # the global m[i] <= M*y[i] (below) will handle upper limit.

# 5.7.2 Activate m_i only when y_i = 1: if y_i == 0 => m_i <= 0; if y_i == 1 => m_i <= M
for i in factories:
    # y_i = 0 -> m_i <= 0  (forces m_i = 0 if no bonus)
    model.addGenConstrIndicator(
        y[i], 0,
        m[i] <= 0,
        name=f"mZeroWhenNoBonus_fac{i+1}"
    )
    # y_i = 1 -> m_i <= M
    model.addGenConstrIndicator(
        y[i], 1,
        m[i] <= M,
        name=f"mUpperWhenBonus_fac{i+1}"
    )

# =========================
# 6. Optimize
# =========================

model.Params.OutputFlag = 0  # Set to 1 if you want detailed solver log
model.optimize()

# =========================
# 7. Print Results
# =========================

if model.status == GRB.OPTIMAL:
    print(f"Optimal objective value (total cost minus bonus): {model.ObjVal:.4f}")

    print("\nAssignment decisions (factory -> orders):")
    for i in factories:
        assigned_orders = []
        for j in orders:
            if Table_1_Cost[i][j] is not None and x[i, j].X > 0.5:
                assigned_orders.append(j + 1)  # convert to 1-based for display
        print(f"Factory {i+1}: orders {assigned_orders}, y={int(y[i].X)}, m={m[i].X:.4f}")

    # Compute the raw total assignment cost and bonus explicitly
    total_assignment_cost = sum(
        Table_1_Cost[i][j] * x[i, j].X
        for i in factories for j in orders
        if Table_1_Cost[i][j] is not None
    )
    total_bonus = bonus_rate * sum(y[i].X * m[i].X for i in factories)

    print(f"\nTotal assignment cost (before bonus): {total_assignment_cost:.4f}")
    print(f"Total performance bonus: {total_bonus:.4f}")
    print(f"Total cost after bonus (objective): {model.ObjVal:.4f}")

    # The question asks: "please give the calculated cost."
    # Interpret this as the optimized total cost after bonus (the objective value).
    final_answer_value = model.ObjVal
else:
    final_answer_value = float('nan')
    print("No optimal solution found.")

# Final answer output in required format
print(f"FinalAnswer=【{final_answer_value}】")