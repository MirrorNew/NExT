import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and any other necessary packages.
# (Already imported above)

# 2. Define all parameter matrices and data inputs.
# Parameters strictly from the Parameters List provided.
number_of_factories = 5
total_orders = 12
standard_orders = 9
joint_orders = 3
factories_per_joint_order = 2
min_orders_per_factory = 1
max_orders_per_factory = 4
bonus_rate = 0.1
Table_1_Cost = [
    [9, 2, 7, 8, None, 6, 5, 4, 3, 11.2, 9.2, 10], 
    [6, 4, 3, None, None, 5, 7, 8, 9, 11.3, 11.4, 11.1], 
    [5, 8, 1, 8, None, 4, 6, 7, 2, 12.0, 10.4, 12.4], 
    [7, 6, 9, 4, 3, 2, 5, 8, 7, 9.9, 10.8, 12.2], 
    [8, 5, 6, 4, 9, 7, 3, 2, 1, 11.4, 10.6, 13.1]
]

# 3. Create Gurobi model.
model = gp.Model("Haitong_Manufacturing")

# 4. Create decision variables.
# x[i, j]: Binary, 1 if factory i is assigned order j
x = {}
for i in range(number_of_factories):
    for j in range(total_orders):
        if Table_1_Cost[i][j] is not None:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

# y[i]: Binary, 1 if factory i qualifies for bonus (>= 3 orders)
y = {}
for i in range(number_of_factories):
    y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")

# m[i]: Continuous, the minimum cost among orders assigned to factory i (if y[i]=1)
m = {}
for i in range(number_of_factories):
    m[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"m_{i}")

# 5. Set up the objective function.
# Minimize Total Cost - Performance Bonus
# Total Cost = Sum(c_ij * x_ij)
# Performance Bonus = Sum(0.1 * m_i)
obj_expr = gp.LinExpr()
for i in range(number_of_factories):
    for j in range(total_orders):
        if Table_1_Cost[i][j] is not None:
            obj_expr += Table_1_Cost[i][j] * x[i, j]
    obj_expr -= bonus_rate * m[i]

model.setObjective(obj_expr, GRB.MINIMIZE)

# 6. Add all constraints.

# (1) Standard-order assignment: Orders 1-9 (indices 0-8)
# Must be completed by exactly one factory.
for j in range(standard_orders):
    model.addConstr(
        gp.quicksum(x[i, j] for i in range(number_of_factories) if Table_1_Cost[i][j] is not None) == 1,
        name=f"Standard_Order_{j+1}"
    )

# (2) Joint-order assignment: Orders 10-12 (indices 9-11)
# Must be completed by exactly two factories.
for j in range(standard_orders, total_orders):
    model.addConstr(
        gp.quicksum(x[i, j] for i in range(number_of_factories) if Table_1_Cost[i][j] is not None) == factories_per_joint_order,
        name=f"Joint_Order_{j+1}"
    )

# (3) Factory Constraints
for i in range(number_of_factories):
    # Workload expression
    workload = gp.quicksum(x[i, j] for j in range(total_orders) if Table_1_Cost[i][j] is not None)
    
    # Min orders per factory
    model.addConstr(workload >= min_orders_per_factory, name=f"Min_Load_{i+1}")
    
    # Max orders per factory
    model.addConstr(workload <= max_orders_per_factory, name=f"Max_Load_{i+1}")
    
    # (4) Bonus Qualification Logic (Indicator Constraints)
    # y[i] = 1 <-> workload >= 3
    # If y[i] = 1, then workload >= 3
    model.addGenConstrIndicator(y[i], 1, workload >= 3, name=f"Bonus_Qual_On_{i+1}")
    # If y[i] = 0, then workload <= 2
    model.addGenConstrIndicator(y[i], 0, workload <= 2, name=f"Bonus_Qual_Off_{i+1}")
    
    # (5) Min-cost linearization (m_i)
    # We want m[i] = min(assigned costs) if y[i]=1, else 0.
    # Since we are minimizing (-m[i]), we are maximizing m[i].
    # We bound m[i] from above by assigned costs.
    
    # Case A: If y[i] = 0, no bonus, set m[i] = 0.
    model.addGenConstrIndicator(y[i], 0, m[i] == 0, name=f"Bonus_Base_Zero_{i+1}")
    
    # Case B: m[i] must be <= cost of any assigned order.
    # Only active if x[i, j] = 1.
    for j in range(total_orders):
        if Table_1_Cost[i][j] is not None:
            model.addGenConstrIndicator(
                x[i, j], 1, m[i] <= Table_1_Cost[i][j], 
                name=f"Min_Cost_Bound_{i+1}_{j+1}"
            )

# 7. Solve the model and print results.
model.optimize()

if model.Status == GRB.OPTIMAL:
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("FinalAnswer=【No Solution】")