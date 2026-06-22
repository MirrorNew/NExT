import gurobipy as gp
from gurobipy import GRB

# ========== 1. Parameters ==========
num_factories = 5
num_orders = 12
standard_orders = 9
joint_orders = 3
factories_per_joint = 2
min_orders = 1
max_orders = 4
bonus_threshold = 3
bonus_rate = 0.1

# Cost matrix from Table_1_Cost
cost = [
    [9, 2, 7, 8, None, 6, 5, 4, 3, 11.2, 9.2, 10],
    [6, 4, 3, None, None, 5, 7, 8, 9, 11.3, 11.4, 11.1],
    [5, 8, 1, 8, None, 4, 6, 7, 2, 12.0, 10.4, 12.4],
    [7, 6, 9, 4, 3, 2, 5, 8, 7, 9.9, 10.8, 12.2],
    [8, 5, 6, 4, 9, 7, 3, 2, 1, 11.4, 10.6, 13.1]
]

# Find M (big number) for constraints
M = 0
for i in range(num_factories):
    for j in range(num_orders):
        if cost[i][j] is not None:
            M = max(M, cost[i][j])
M = M + 10  # Add a safety margin

# ========== 2. Create Model ==========
model = gp.Model("OrderAssignment")

# ========== 3. Decision Variables ==========
x = {}
for i in range(num_factories):
    for j in range(num_orders):
        if cost[i][j] is not None:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        else:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}", ub=0)  # fixed to 0

y = {}
for i in range(num_factories):
    y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")

m = {}
for i in range(num_factories):
    m[i] = model.addVar(lb=0, ub=M, vtype=GRB.CONTINUOUS, name=f"m_{i}")

model.update()

# ========== 4. Constraints ==========
# Standard orders: exactly one factory each
for j in range(standard_orders):
    model.addConstr(
        gp.quicksum(x[i, j] for i in range(num_factories) if cost[i][j] is not None) == 1,
        name=f"standard_order_{j}"
    )

# Joint orders: exactly two factories each
for j in range(standard_orders, num_orders):
    model.addConstr(
        gp.quicksum(x[i, j] for i in range(num_factories) if cost[i][j] is not None) == factories_per_joint,
        name=f"joint_order_{j}"
    )

# Min orders per factory
for i in range(num_factories):
    model.addConstr(
        gp.quicksum(x[i, j] for j in range(num_orders)) >= min_orders,
        name=f"min_orders_factory_{i}"
    )

# Max orders per factory
for i in range(num_factories):
    model.addConstr(
        gp.quicksum(x[i, j] for j in range(num_orders)) <= max_orders,
        name=f"max_orders_factory_{i}"
    )

# Link y_i to order count (if >=3 orders then y_i=1)
for i in range(num_factories):
    # y_i = 1 -> sum x_ij >= 3
    model.addGenConstrIndicator(
        y[i], 1,
        gp.quicksum(x[i, j] for j in range(num_orders)) >= bonus_threshold,
        name=f"y_lower_{i}"
    )
    # y_i = 0 -> sum x_ij <= 2
    model.addGenConstrIndicator(
        y[i], 0,
        gp.quicksum(x[i, j] for j in range(num_orders)) <= 2,
        name=f"y_upper_{i}"
    )

# Define m_i as minimum cost among assigned orders
for i in range(num_factories):
    for j in range(num_orders):
        if cost[i][j] is not None:
            # If x_ij = 1, then m_i <= c_ij
            model.addGenConstrIndicator(
                x[i, j], 1,
                m[i] <= cost[i][j],
                name=f"min_cost_{i}_{j}"
            )
    # m_i <= M * y_i (only active if y_i=1)
    model.addGenConstrIndicator(
        y[i], 0,
        m[i] <= 0,  # if y_i=0, m_i must be 0 (since it's not eligible for bonus)
        name=f"m_activation_{i}"
    )

# ========== 5. Objective ==========
total_cost = gp.quicksum(
    cost[i][j] * x[i, j]
    for i in range(num_factories)
    for j in range(num_orders)
    if cost[i][j] is not None
)

bonus = gp.quicksum(bonus_rate * y[i] * m[i] for i in range(num_factories))

model.setObjective(total_cost - bonus, GRB.MINIMIZE)

# ========== 6. Solve ==========
model.optimize()

# ========== 7. Results ==========
if model.status == GRB.OPTIMAL:
    print("Optimal solution found.")
    print(f"Total cost (before bonus): {total_cost.getValue():.2f}")
    print(f"Total bonus: {bonus.getValue():.2f}")
    print(f"Net cost: {model.ObjVal:.2f}")
    
    # Assignment details
    for i in range(num_factories):
        assigned = [j for j in range(num_orders) if x[i, j].X > 0.5]
        print(f"Factory {i+1}: orders {[order+1 for order in assigned]}, "
              f"count={len(assigned)}, y={y[i].X}, m={m[i].X:.2f}")
    
    # Final answer (the question asks for the calculated cost)
    final_cost = model.ObjVal
    print(f"FinalAnswer=【{final_cost:.2f}】")
else:
    print("No optimal solution found.")
    print(f"FinalAnswer=【None】")