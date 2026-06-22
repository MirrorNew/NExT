import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
# Table C-39: Customer flow (thousand people/day) between commodity categories
Table_1_C_39 = [
    [0, 5, 2, 7],
    [5, 0, 3, 8],
    [2, 3, 0, 3],
    [7, 8, 3, 0]
]

# Table C-40: Average distance customers move between blocks (m)
# Blocks I, II, III, IV mapped to indices 0, 1, 2, 3
Table_2_C_40 = [
    [0, 20, 37.5, 42.5],
    [20, 0, 32.5, 25],
    [37.5, 32.5, 0, 30],
    [42.5, 25, 30, 0]
]

num_categories = 4
num_blocks = 4

# Create the Gurobi model
model = gp.Model("Supermarket_Layout_Optimization")

# 2. Identify any function expressions that require auxiliary substitution variables,
# and use "model.Params.NonConvex = 2" as the problem involves products of binary variables.
model.Params.NonConvex = 2

# 3. Create decision variables.
# x[i, k] = 1 if commodity category i is assigned to block k, 0 otherwise.
x = model.addVars(num_categories, num_blocks, vtype=GRB.BINARY, name="x")

# 4. Create auxiliary substitution variables.
# The values of these auxiliary variables range from negative infinity to positive infinity.
# y[i, j, k, l] represents the substitution for the product x[i, k] * x[j, l].
y = {}
for i in range(num_categories):
    for j in range(num_categories):
        for k in range(num_blocks):
            for l in range(num_blocks):
                y_name = f"y_{i}_{j}_{k}_{l}"
                y[i, j, k, l] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=y_name)
                # Substitution constraint for the quadratic terms.
                model.addConstr(y[i, j, k, l] == x[i, k] * x[j, l])

# 5. Set up the objective function.
# The objective is to minimize the total round trip distance that customers walk.
# Per the analysis of the previous failed attempts, the double summation over a symmetric
# flow matrix C_ij and symmetric distance matrix D_kl already accounts for movement
# in both directions (round trip). We multiply by 1000 to convert flow from thousands to individuals.
objective = gp.quicksum(1000 * Table_1_C_39[i][j] * Table_2_C_40[k][l] * y[i, j, k, l]
                        for i in range(num_categories)
                        for j in range(num_categories)
                        for k in range(num_blocks)
                        for l in range(num_blocks))

model.setObjective(objective, GRB.MINIMIZE)

# 6. Add all constraints.
# Category assignment: Each category must be assigned to exactly one block.
for i in range(num_categories):
    model.addConstr(gp.quicksum(x[i, k] for k in range(num_blocks)) == 1, name=f"AssignCategory_{i}")

# Block occupancy: Each block must be occupied by exactly one category.
for k in range(num_blocks):
    model.addConstr(gp.quicksum(x[i, k] for i in range(num_categories)) == 1, name=f"OccupancyBlock_{k}")

# 7. Solve the model and print the result.
model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("Optimization was not successful.")