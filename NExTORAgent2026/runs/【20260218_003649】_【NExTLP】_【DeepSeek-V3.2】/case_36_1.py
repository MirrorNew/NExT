import gurobipy as gp
from gurobipy import GRB

# 1. Define parameters from the provided list
# Table C-39: Customer flow (thousand people/day) between categories
# Index mapping: 0->tobacco_and_alcohol, 1->vegetables_and_fruits, 2->grain_and_non_staple_food, 3->daily_necessities
c = [[0, 5, 2, 7],
     [5, 0, 3, 8],
     [2, 3, 0, 3],
     [7, 8, 3, 0]]

# Table C-40: Average distance between blocks (m)
# Index mapping: 0->I, 1->II, 2->III, 3->IV
d = [[0, 20, 37.5, 42.5],
     [20, 0, 32.5, 25],
     [37.5, 32.5, 0, 30],
     [42.5, 25, 30, 0]]

# Number of categories and blocks
num_categories = 4
num_blocks = 4
category_indices = range(num_categories)
block_indices = range(num_blocks)

# 2. Create model
model = gp.Model("SupermarketLayout_QAP")

# 3. Create decision variables
x = model.addVars(num_categories, num_blocks, vtype=GRB.BINARY, name="x")

# 4. Create auxiliary variables y_{ijkl} = x_{ik} * x_{jl} for i<j and k≠l
y = {}
for i in range(num_categories):
    for j in range(i+1, num_categories):
        for k in range(num_blocks):
            for l in range(num_blocks):
                if k != l:
                    y[(i,j,k,l)] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, 
                                                name=f"y_{i}_{j}_{k}_{l}")

# 5. Set objective function
# Minimize total round-trip distance: Z = 2 * sum_{i<j} sum_{k≠l} c_{ij} * d_{kl} * y_{ijkl}
obj_expr = gp.QuadExpr()
for i in range(num_categories):
    for j in range(i+1, num_categories):
        for k in range(num_blocks):
            for l in range(num_blocks):
                if k != l:
                    obj_expr += 2 * c[i][j] * d[k][l] * y[(i,j,k,l)]
model.setObjective(obj_expr, GRB.MINIMIZE)

# 6. Add constraints

# 6.1 Each category assigned to exactly one block
for i in category_indices:
    model.addConstr(gp.quicksum(x[i,k] for k in block_indices) == 1, 
                    name=f"CategoryAssign_{i}")

# 6.2 Each block receives exactly one category
for k in block_indices:
    model.addConstr(gp.quicksum(x[i,k] for i in category_indices) == 1, 
                    name=f"BlockAssign_{k}")

# 6.3 Linearization constraints for y_{ijkl} = x_{ik} * x_{jl}
for i in range(num_categories):
    for j in range(i+1, num_categories):
        for k in range(num_blocks):
            for l in range(num_blocks):
                if k != l:
                    # y_{ijkl} ≤ x_{ik}
                    model.addConstr(y[(i,j,k,l)] <= x[i,k], 
                                    name=f"Lin1_{i}_{j}_{k}_{l}")
                    # y_{ijkl} ≤ x_{j,l}
                    model.addConstr(y[(i,j,k,l)] <= x[j,l], 
                                    name=f"Lin2_{i}_{j}_{k}_{l}")
                    # y_{ijkl} ≥ x_{ik} + x_{jl} - 1
                    model.addConstr(y[(i,j,k,l)] >= x[i,k] + x[j,l] - 1, 
                                    name=f"Lin3_{i}_{j}_{k}_{l}")

# 7. Solve the model
model.Params.NonConvex = 2  # Allow quadratic terms in constraints
model.optimize()

# 8. Print results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"Objective value (total round-trip distance): {model.objVal:.2f} thousand person-meters/day")
    
    # Print assignment
    block_names = ['I', 'II', 'III', 'IV']
    category_names = ['Tobacco and alcohol', 'Vegetables and fruits', 
                      'Grain and non-staple food', 'Daily necessities']
    
    print("\nOptimal assignment:")
    for i in category_indices:
        for k in block_indices:
            if x[i,k].X > 0.5:
                print(f"{category_names[i]} -> Block {block_names[k]}")
    
    # Calculate and print the distance as required
    total_distance = model.objVal
    print(f"\nFinalAnswer=【{total_distance:.2f}】")
else:
    print(f"Optimization failed. Status: {model.status}")
    print(f"FinalAnswer=【Optimization failed】")