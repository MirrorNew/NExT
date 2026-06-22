import gurobipy as gp
from gurobipy import GRB

# Initialize the model
model = gp.Model("Supermarket_Layout_Optimization")

# 1. Define Parameters
# Customer flow (thousand people/day) between goods categories - Table C-39
# Indices: 0: Tobacco/Alcohol, 1: Veg/Fruits, 2: Grain/Non-staple, 3: Daily necessities
flows = [
    [0, 5, 2, 7],
    [5, 0, 3, 8],
    [2, 3, 0, 3],
    [7, 8, 3, 0]
]

# Average distance (m) between blocks - Table C-40
# Indices: 0: Block I, 1: Block II, 2: Block III, 3: Block IV
distances = [
    [0, 20, 37.5, 42.5],
    [20, 0, 32.5, 25],
    [37.5, 32.5, 0, 30],
    [42.5, 25, 30, 0]
]

num_goods = 4
num_blocks = 4

# 2. Create Decision Variables
# x[i,k] = 1 if good i is assigned to block k
x = model.addVars(num_goods, num_blocks, vtype=GRB.BINARY, name="x")

# 3. Set up the Objective Function
# Objective: Min Z = 2 * sum(F_ij * D_kl * x_ik * x_jl)
# We build the quadratic expression term by term.
obj_expr = 0
for i in range(num_goods):
    for j in range(num_goods):
        for k in range(num_blocks):
            for l in range(num_blocks):
                # Add term only if coefficients are non-zero
                if flows[i][j] > 0 and distances[k][l] > 0:
                    obj_expr += flows[i][j] * distances[k][l] * x[i, k] * x[j, l]

# The problem requires calculating total round trip distance (factor of 2)
model.setObjective(2 * obj_expr, GRB.MINIMIZE)

# 4. Add Constraints
# Constraint 1: Each commodity category assigned to exactly one block
model.addConstrs((x.sum(i, '*') == 1 for i in range(num_goods)), name="Assign_Good")

# Constraint 2: Each block occupied by exactly one commodity category
model.addConstrs((x.sum('*', k) == 1 for k in range(num_blocks)), name="Occup_Block")

# 5. Solve the Model
# Set NonConvex parameter to 2 to allow Gurobi to solve the MIQP (Mixed Integer Quadratic Programming)
model.Params.NonConvex = 2
model.optimize()

# 6. Output Results
if model.Status == GRB.OPTIMAL:
    # Output the optimized objective value as required
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("FinalAnswer=【No Solution Found】")