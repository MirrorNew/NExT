import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs
# Parameters strictly from the Parameters List provided.
number_of_factories = 5
total_orders = 12
standard_orders = 9
joint_orders = 3
factories_per_joint_order = 2
min_orders_per_factory = 1
max_orders_per_factory = 4
bonus_rate = 0.1

# Cost Matrix (None indicates unqualified)
Table_1_Cost = [
    [9, 2, 7, 8, None, 6, 5, 4, 3, 11.2, 9.2, 10], 
    [6, 4, 3, None, None, 5, 7, 8, 9, 11.3, 11.4, 11.1], 
    [5, 8, 1, 8, None, 4, 6, 7, 2, 12.0, 10.4, 12.4], 
    [7, 6, 9, 4, 3, 2, 5, 8, 7, 9.9, 10.8, 12.2], 
    [8, 5, 6, 4, 9, 7, 3, 2, 1, 11.4, 10.6, 13.1]
]

# 2. Create Gurobi model
model = gp.Model("Haitong_Manufacturing_Optimization")

# 3. Create decision variables
# x[i, j]: Binary, 1 if factory i is assigned order j
x = {}
for i in range(number_of_factories):
    for j in range(total_orders):
        # Only create variable if factory is qualified
        if Table_1_Cost[i][j] is not None:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

# y[i]: Binary, 1 if factory i qualifies for bonus (>= 3 orders)
y = {}
for i in range(number_of_factories):
    y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")

# m[i]: Continuous, represents the base for the bonus (min cost of assigned orders)
# We minimize -m[i] (maximize m[i]) subject to m[i] <= assigned_costs.
m = {}
for i in range(number_of_factories):
    m[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"m_{i}")

# 4. Set up the objective function
# Minimize Total Cost - Performance Bonus
# Bonus = 0.1 * m_i
obj_expr = gp.LinExpr()

# Add costs
for i in range(number_of_factories):
    for j in range(total_orders):
        if Table_1_Cost[i][j] is not None:
            obj_expr += Table_1_Cost[i][j] * x[i, j]

# Subtract bonus
for i in range(number_of_factories):
    obj_expr -= bonus_rate * m[i]

model.setObjective(obj_expr, GRB.MINIMIZE)

# 5. Add all constraints

# (1) Standard-order assignment (Orders 1-9, indices 0-8)
# Must be assigned to exactly 1 factory
for j in range(standard_orders):
    model.addConstr(
        gp.quicksum(x[i, j] for i in range(number_of_factories) if Table_1_Cost[i][j] is not None) == 1,
        name=f"Standard_Order_{j+1}"
    )

# (2) Joint-order assignment (Orders 10-12, indices 9-11)
# Must be assigned to exactly 2 factories
for j in range(standard_orders, total_orders):
    model.addConstr(
        gp.quicksum(x[i, j] for i in range(number_of_factories) if Table_1_Cost[i][j] is not None) == factories_per_joint_order,
        name=f"Joint_Order_{j+1}"
    )

# (3) Factory Constraints (Workload & Bonus Logic)
for i in range(number_of_factories):
    # Calculate workload (number of orders assigned to factory i)
    workload = gp.quicksum(x[i, j] for j in range(total_orders) if Table_1_Cost[i][j] is not None)
    
    # Min orders per factory constraint
    model.addConstr(workload >= min_orders_per_factory, name=f"Min_Load_{i+1}")
    
    # Max orders per factory constraint
    model.addConstr(workload <= max_orders_per_factory, name=f"Max_Load_{i+1}")
    
    # Bonus Qualification Logic: y[i] = 1 <-> workload >= 3
    # If y[i]=1, then workload >= 3
    model.addGenConstrIndicator(y[i], 1, workload >= 3, name=f"Bonus_Qual_On_{i+1}")
    # If y[i]=0, then workload <= 2
    model.addGenConstrIndicator(y[i], 0, workload <= 2, name=f"Bonus_Qual_Off_{i+1}")
    
    # Min-cost linearization for bonus calculation
    # Logic: 
    # If eligible (y=1), m[i] = min(cost of assigned orders). 
    # Since we maximize m[i] (via minimizing negative objective), we just need upper bounds: m[i] <= cost if assigned.
    # If not eligible (y=0), m[i] = 0.
    
    # Case A: Not eligible -> m[i] = 0
    model.addGenConstrIndicator(y[i], 0, m[i] == 0, name=f"Bonus_Base_Zero_{i+1}")
    
    # Case B: Eligible (implicitly handled by maximization). Add upper bounds for assigned orders.
    # If x[i,j]=1, then m[i] <= cost[i,j]. 
    # If x[i,j]=0, this constraint is inactive (handled by indicator).
    for j in range(total_orders):
        if Table_1_Cost[i][j] is not None:
            model.addGenConstrIndicator(
                x[i, j], 1, m[i] <= Table_1_Cost[i][j], 
                name=f"Min_Cost_Bound_{i+1}_{j+1}"
            )

# 6. Solve the model and print results
model.optimize()

if model.Status == GRB.OPTIMAL:
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("FinalAnswer=【No Solution】")