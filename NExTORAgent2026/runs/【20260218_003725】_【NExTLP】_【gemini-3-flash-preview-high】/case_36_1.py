import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and define all parameter matrices and data inputs.
# Data from the problem description
residential_communities_count = 12
schools_count = 3
radius_m = 500
zones_count = 4
living_circle_time_min = 15
goods_categories = [None, 'tobacco_and_alcohol', 'vegetables_and_fruits', 'grain_and_non_staple_food', 'daily_necessities']
floor_level = 1
number_doors = 2
number_walkways = 2
block_names = ['I', 'II', 'III', 'IV']

# Table C-39 Customer flow (thousand people/day)
Table_1_C_39 = [
    [0, 5, 2, 7],
    [5, 0, 3, 8],
    [2, 3, 0, 3],
    [7, 8, 3, 0]
]

# Table C-40 Average distance customers move between blocks (m)
Table_2_C_40 = [
    [0, 20, 37.5, 42.5],
    [20, 0, 32.5, 25],
    [37.5, 32.5, 0, 30],
    [42.5, 25, 30, 0]
]

num_categories = 4
num_blocks = 4

# Create Gurobi model
model = gp.Model("Supermarket_Layout_Optimization")

# Identify any function expressions that require auxiliary substitution variables,
# and use "model.Params.NonConvex = 2" as the model involves quadratic constraints for binary variables.
model.Params.NonConvex = 2

# 2. Create decision variables.
# x[i, k] = 1 if commodity category i is assigned to block k, 0 otherwise
x = model.addVars(num_categories, num_blocks, vtype=GRB.BINARY, name="x")

# 3. Create auxiliary substitution variables.
# y[i, j, k, l] representing the product of x[i, k] and x[j, l]
# Per coding instructions, auxiliary variables range from negative infinity to positive infinity.
y = {}
for i in range(num_categories):
    for j in range(num_categories):
        for k in range(num_blocks):
            for l in range(num_blocks):
                y_name = f"y_{i}_{j}_{k}_{l}"
                y[i, j, k, l] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=y_name)

# 4. Set up the objective function.
# Min Z = 2 * sum_{i,j,k,l} (1000 * C_ij) * D_kl * x_ik * x_jl
# C_ij is the customer flow in thousands of people, so we multiply by 1000.
# The factor of 2 accounts for the round trip distance.
objective = gp.quicksum(2 * 1000 * Table_1_C_39[i][j] * Table_2_C_40[k][l] * y[i, j, k, l]
                        for i in range(num_categories)
                        for j in range(num_categories)
                        for k in range(num_blocks)
                        for l in range(num_blocks))

model.setObjective(objective, GRB.MINIMIZE)

# 5. Add all constraints.
# Category assignment: Each category must be assigned to exactly one block.
for i in range(num_categories):
    model.addConstr(gp.quicksum(x[i, k] for k in range(num_blocks)) == 1, name=f"Category_Assignment_{i}")

# Block occupancy: Each block must be occupied by exactly one category.
for k in range(num_blocks):
    model.addConstr(gp.quicksum(x[i, k] for i in range(num_categories)) == 1, name=f"Block_Occupancy_{k}")

# Substitution constraints for auxiliary variables: y[i, j, k, l] = x[i, k] * x[j, l]
for i in range(num_categories):
    for j in range(num_categories):
        for k in range(num_blocks):
            for l in range(num_blocks):
                model.addConstr(y[i, j, k, l] == x[i, k] * x[j, l], name=f"Substitution_{i}_{j}_{k}_{l}")

# 6. Solve the model and print results.
model.optimize()

if model.status == GRB.OPTIMAL:
    # Print the objective value as the FinalAnswer.
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found.")