import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs
num_factories = 5
total_orders = 12
standard_orders = 9
joint_orders = 3
min_orders_per_factory = 1
max_orders_per_factory = 4
bonus_threshold = 3
bonus_rate = 0.1

# Table_1_Cost provided as a list of lists. None represents the factory is not qualified.
costs = [
    [9, 2, 7, 8, None, 6, 5, 4, 3, 11.2, 9.2, 10],
    [6, 4, 3, None, None, 5, 7, 8, 9, 11.3, 11.4, 11.1],
    [5, 8, 1, 8, None, 4, 6, 7, 2, 12.0, 10.4, 12.4],
    [7, 6, 9, 4, 3, 2, 5, 8, 7, 9.9, 10.8, 12.2],
    [8, 5, 6, 4, 9, 7, 3, 2, 1, 11.4, 10.6, 13.1]
]

# 2. Create the model
model = gp.Model("Haitong_Manufacturing_Optimization")

# 3. Create decision variables
# x[i, j] is 1 if factory i is assigned to order j, 0 otherwise
x = {}
for i in range(num_factories):
    for j in range(total_orders):
        if costs[i][j] is not None:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        else:
            # If not qualified, x[i, j] must be 0
            x[i, j] = model.addVar(vtype=GRB.BINARY, lb=0, ub=0, name=f"x_{i}_{j}")

# y[i] is 1 if factory i undertakes 3 or more orders, 0 otherwise
y = model.addVars(num_factories, vtype=GRB.BINARY, name="y")

# m[i] is a helper variable representing the minimum cost among orders assigned to factory i (if y[i] is 1)
m = model.addVars(num_factories, vtype=GRB.CONTINUOUS, lb=0, name="m")

# 4. Set up the objective function
# Minimize: (Total assignment cost) - (Performance bonus)
total_assignment_cost = gp.quicksum(costs[i][j] * x[i, j] for i in range(num_factories) for j in range(total_orders) if costs[i][j] is not None)
total_performance_bonus = gp.quicksum(bonus_rate * m[i] for i in range(num_factories))
model.setObjective(total_assignment_cost - total_performance_bonus, GRB.MINIMIZE)

# 5. Add all constraints

# (1) Standard-order assignment: each standard order must be assigned to exactly 1 factory
for j in range(standard_orders):
    model.addConstr(gp.quicksum(x[i, j] for i in range(num_factories)) == 1, name=f"StandardAssignment_{j}")

# (2) Joint-order assignment: each joint order must be assigned to exactly 2 factories
for j in range(standard_orders, total_orders):
    model.addConstr(gp.quicksum(x[i, j] for i in range(num_factories)) == 2, name=f"JointAssignment_{j}")

# (3) Factory order limits: each factory must undertake between 1 and 4 orders
for i in range(num_factories):
    order_count = gp.quicksum(x[i, j] for j in range(total_orders))
    model.addConstr(order_count >= min_orders_per_factory, name=f"MinOrders_F{i+1}")
    model.addConstr(order_count <= max_orders_per_factory, name=f"MaxOrders_F{i+1}")

# (4) Bonus qualification indicator y[i]
# Indicator constraints for bonus condition: y[i]=1 iff order_count >= 3
for i in range(num_factories):
    order_count = gp.quicksum(x[i, j] for j in range(total_orders))
    model.addGenConstrIndicator(y[i], 1, order_count >= bonus_threshold, name=f"BonusInd_1_F{i+1}")
    model.addGenConstrIndicator(y[i], 0, order_count <= (bonus_threshold - 1), name=f"BonusInd_0_F{i+1}")

# (5) Bonus calculation: m[i] should be the minimum cost of the items assigned to factory i
M = 30  # A large constant (any value greater than the max possible single order cost is sufficient)
for i in range(num_factories):
    # m[i] is limited by the costs of all orders assigned to factory i
    for j in range(total_orders):
        if costs[i][j] is not None:
            # If x[i, j] = 1, m[i] <= costs[i][j]
            # If x[i, j] = 0, m[i] <= M (no real restriction)
            model.addConstr(m[i] <= costs[i][j] * x[i, j] + M * (1 - x[i, j]), name=f"MinCostRef_F{i+1}_O{j+1}")
    
    # If the factory does not qualify for bonus (y[i]=0), m[i] must be 0
    model.addConstr(m[i] <= M * y[i], name=f"BonusActivation_F{i+1}")

# 6. Solve the model and print results
model.optimize()

if model.status == GRB.OPTIMAL:
    final_cost = model.objVal
    print(f"FinalAnswer=【{final_cost}】")
else:
    print("No optimal solution was found.")