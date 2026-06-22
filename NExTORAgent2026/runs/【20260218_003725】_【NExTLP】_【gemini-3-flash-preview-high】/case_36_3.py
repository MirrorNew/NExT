import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
Table_1_C_39 = [
    [0, 5, 2, 7],
    [5, 0, 3, 8],
    [2, 3, 0, 3],
    [7, 8, 3, 0]
]

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

# Identify any function expressions that require auxiliary substitution variables,
# and use "model.Params.NonConvex = 2" as needed.
model.Params.NonConvex = 2

# 2. Create decision variables.
# x[i, k] = 1 if commodity category i is assigned to block k; = 0 otherwise.
x = model.addVars(num_categories, num_blocks, vtype=GRB.BINARY, name="x")

# 3. Create any auxiliary substitution variables.
# Substitution for y[i, j, k, l] = x[i, k] * x[j, l]
# Per instructions, the values of these auxiliary variables range from -infinity to infinity.
y = {}
for i in range(num_categories):
    for j in range(num_categories):
        for k in range(num_blocks):
            for l in range(num_blocks):
                y_name = f"y_{i}_{j}_{k}_{l}"
                y[i, j, k, l] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=y_name)

# 4. Set up the objective function.
# Min Z = 2 · ∑_{i} ∑_{j} ∑_{k} ∑_{l} (1000 · C_{ij}) · D_{kl} · x_{ik} · x_{jl}
# Multiplying flow by 1000 as Table C-39 is in thousands.
objective = gp.quicksum(2 * 1000 * Table_1_C_39[i][j] * Table_2_C_40[k][l] * y[i, j, k, l]
                        for i in range(num_categories)
                        for j in range(num_categories)
                        for k in range(num_blocks)
                        for l in range(num_blocks))

model.setObjective(objective, GRB.MINIMIZE)

# 5. Add all constraints.

# Category assignment: Each commodity category i must be assigned to exactly one block k.
for i in range(num_categories):
    model.addConstr(gp.quicksum(x[i, k] for k in range(num_blocks)) == 1, name=f"CategoryAssignment_{i}")

# Block occupancy: Each block k must be occupied by exactly one commodity category i.
for k in range(num_blocks):
    model.addConstr(gp.quicksum(x[i, k] for i in range(num_categories)) == 1, name=f"BlockOccupancy_{k}")

# Link auxiliary substitution variables through quadratic constraints.
for i in range(num_categories):
    for j in range(num_categories):
        for k in range(num_blocks):
            for l in range(num_blocks):
                model.addConstr(y[i, j, k, l] == x[i, k] * x[j, l], name=f"Substitution_{i}_{j}_{k}_{l}")

# 6. Solve the model and print results.
model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"FinalAnswer=【{model.objVal}】")