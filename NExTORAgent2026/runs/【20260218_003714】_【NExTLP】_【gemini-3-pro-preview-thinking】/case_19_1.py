import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs
# Value from Parameters List
costs = [
    [9, 2, 7, 8, None, 6, 5, 4, 3, 11.2, 9.2, 10], 
    [6, 4, 3, None, None, 5, 7, 8, 9, 11.3, 11.4, 11.1], 
    [5, 8, 1, 8, None, 4, 6, 7, 2, 12.0, 10.4, 12.4], 
    [7, 6, 9, 4, 3, 2, 5, 8, 7, 9.9, 10.8, 12.2], 
    [8, 5, 6, 4, 9, 7, 3, 2, 1, 11.4, 10.6, 13.1]
]

num_factories = 5
num_orders = 12
standard_orders_count = 9  # Orders 1-9 (indices 0-8)
joint_orders_count = 3     # Orders 10-12 (indices 9-11)
bonus_rate = 0.1

# 2. Create Gurobi Model
model = gp.Model("Haitong_Manufacturing_Optimization")

# 3. Create decision variables
# x[i,j] = 1 if factory i assigned order j
x = {}
for i in range(num_factories):
    for j in range(num_orders):
        # Create variable only if the factory is qualified (cost is not None)
        if costs[i][j] is not None:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

# y[i] = 1 if factory i qualifies for bonus (>= 3 orders)
y = {}
for i in range(num_factories):
    y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")

# m[i] = minimum cost of assigned orders for factory i (bonus base)
# Bounded below by 0
m = {}
for i in range(num_factories):
    m[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"m_{i}")

# 4. Set up the objective function
# Min Z = Total Cost - Bonus
# Total Cost = sum(c_ij * x_ij)
# Bonus = sum(0.1 * m_i)
# Note: Since we minimize (Cost - 0.1*m_i), we are maximizing m_i.
# The constraints will bound m_i from above by the actual costs.
total_cost = gp.quicksum(costs[i][j] * x[i, j] for i in range(num_factories) for j in range(num_orders) if costs[i][j] is not None)
total_bonus = gp.quicksum(bonus_rate * m[i] for i in range(num_factories))

model.setObjective(total_cost - total_bonus, GRB.MINIMIZE)

# 5. Add all constraints

# (1) Standard-order assignment: Orders 1-9 (indices 0-8) assigned to exactly 1 factory
for j in range(standard_orders_count):
    model.addConstr(gp.quicksum(x[i, j] for i in range(num_factories) if costs[i][j] is not None) == 1, name=f"Standard_Order_{j+1}")

# (2) Joint-order assignment: Orders 10-12 (indices 9-11) assigned to exactly 2 factories
for j in range(standard_orders_count, num_orders):
    model.addConstr(gp.quicksum(x[i, j] for i in range(num_factories) if costs[i][j] is not None) == 2, name=f"Joint_Order_{j+1}")

# (3) Factory workload limits and Bonus Logic
for i in range(num_factories):
    # Calculate workload for factory i
    workload = gp.quicksum(x[i, j] for j in range(num_orders) if costs[i][j] is not None)
    
    # Min orders per factory >= 1
    model.addConstr(workload >= 1, name=f"Min_Load_Factory_{i+1}")
    
    # Max orders per factory <= 4
    model.addConstr(workload <= 4, name=f"Max_Load_Factory_{i+1}")
    
    # Link y variable to workload using Indicator Constraints
    # If y_i = 1, then workload >= 3
    model.addGenConstrIndicator(y[i], 1, workload >= 3, name=f"Bonus_Qualify_{i+1}")
    # If y_i = 0, then workload <= 2
    model.addGenConstrIndicator(y[i], 0, workload <= 2, name=f"Bonus_Disqualify_{i+1}")
    
    # Link m variable (Bonus Base)
    # Case A: If y_i = 0 (no bonus), m_i must be 0
    model.addGenConstrIndicator(y[i], 0, m[i] == 0, name=f"Bonus_Base_Zero_{i+1}")
    
    # Case B: If x_ij = 1, then m_i <= c_ij
    # This applies regardless of y, but effectively bounds m_i when y=1.
    # Since we maximize m_i (via minimizing -m_i), m_i will equal min(assigned costs) if y=1.
    for j in range(num_orders):
        if costs[i][j] is not None:
            model.addGenConstrIndicator(x[i, j], 1, m[i] <= costs[i][j], name=f"Min_Cost_Bound_{i+1}_{j+1}")

# 6. Solve the model and print results
model.optimize()

if model.Status == GRB.OPTIMAL:
    # Print the final calculated cost (Objective value)
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("FinalAnswer=【No Solution】")