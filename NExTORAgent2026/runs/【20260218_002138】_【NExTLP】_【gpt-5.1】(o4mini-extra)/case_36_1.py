import gurobipy as gp
from gurobipy import GRB

# ==============================
# 1. Define parameter data
# ==============================

# Parameters List (only using given values)
residential_communities_count = 12
schools_count = 3
radius_m = 500
zones_count = 4
living_circle_time_min = 15
goods_categories = [None, 'tobacco_and_alcohol', 'vegetables_and_fruits',
                    'grain_and_non_staple_food', 'daily_necessities']
floor_level = 1
number_doors = 2
number_walkways = 2
block_names = ['I', 'II', 'III', 'IV']

# Table_1_C_39: customer flow between commodity categories (thousand people/day)
Table_1_C_39 = [
    [0, 5, 2, 7],
    [5, 0, 3, 8],
    [2, 3, 0, 3],
    [7, 8, 3, 0]
]

# Table_2_C_40: distance between blocks (m)
Table_2_C_40 = [
    [0, 20, 37.5, 42.5],
    [20, 0, 32.5, 25],
    [37.5, 32.5, 0, 30],
    [42.5, 25, 30, 0]
]

# Index sets for goods (1..4) and blocks (1..4)
goods_index = range(4)   # 0..3 represent categories 1..4
blocks_index = range(4)  # 0..3 represent blocks I..IV

# ==============================
# 2. Create model
# ==============================
model = gp.Model("Supermarket_Block_Assignment")

# ==============================
# 3. Decision variables x_{ik}
# ==============================
# x[i,k] = 1 if goods category i is assigned to block k
x = model.addVars(goods_index, blocks_index, vtype=GRB.BINARY, name="x")

# ==============================
# 4. Auxiliary variables z_{ij,kl} = x_{ik} * x_{jl}
# ==============================
# As per advice, introduce linearization variables for the bilinear term
z = model.addVars(
    goods_index, goods_index, blocks_index, blocks_index,
    vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="z"
)

# ==============================
# 5. Assignment constraints
# ==============================

# Each goods category assigned to exactly one block
for i in goods_index:
    model.addConstr(gp.quicksum(x[i, k] for k in blocks_index) == 1,
                    name=f"assign_good_{i}")

# Each block has exactly one goods category
for k in blocks_index:
    model.addConstr(gp.quicksum(x[i, k] for i in goods_index) == 1,
                    name=f"assign_block_{k}")

# ==============================
# 6. Linearization constraints for z_{ij,kl}
# ==============================
# z[i,j,k,l] = x[i,k] * x[j,l] via standard McCormick for binaries
for i in goods_index:
    for j in goods_index:
        for k in blocks_index:
            for l in blocks_index:
                # z <= x_{ik}
                model.addConstr(z[i, j, k, l] <= x[i, k],
                                name=f"z_le_xik_{i}_{j}_{k}_{l}")
                # z <= x_{jl}
                model.addConstr(z[i, j, k, l] <= x[j, l],
                                name=f"z_le_xjl_{i}_{j}_{k}_{l}")
                # z >= x_{ik} + x_{jl} - 1
                model.addConstr(
                    z[i, j, k, l] >= x[i, k] + x[j, l] - 1,
                    name=f"z_ge_xik_xjl_minus1_{i}_{j}_{k}_{l}"
                )

# ==============================
# 7. Objective function
# ==============================
# Minimize total round-trip walking distance:
# Min Z = 2 * sum_{i,j,k,l} c_ij * d_kl * z_{ij,kl}
obj_expr = gp.quicksum(
    2.0 * Table_1_C_39[i][j] * Table_2_C_40[k][l] * z[i, j, k, l]
    for i in goods_index
    for j in goods_index
    for k in blocks_index
    for l in blocks_index
)

model.setObjective(obj_expr, GRB.MINIMIZE)

# ==============================
# 8. Optimize
# ==============================
model.optimize()

# ==============================
# 9. Retrieve and print results
# ==============================
if model.Status == GRB.OPTIMAL:
    print(f"Optimal objective (total round-trip distance, meter-thousand-people/day units): {model.ObjVal}")

    # Mapping indices back to names
    goods_names = {
        0: 'tobacco_and_alcohol',
        1: 'vegetables_and_fruits',
        2: 'grain_and_non_staple_food',
        3: 'daily_necessities'
    }

    block_label = {
        0: 'I',
        1: 'II',
        2: 'III',
        3: 'IV'
    }

    print("\nOptimal assignment of goods categories to blocks:")
    assignment = {}
    for i in goods_index:
        for k in blocks_index:
            if x[i, k].X > 0.5:
                assignment[goods_names[i]] = block_label[k]
                print(f"  {goods_names[i]} -> Block {block_label[k]}")

    # The question asks: "What is the distance? (Calculate the total round trip distance)"
    # So the answer is the optimal objective value.
    total_round_trip_distance = model.ObjVal

    # Final required output format
    print(f"FinalAnswer=【{total_round_trip_distance}】")
else:
    print("No optimal solution found.")
    # Still print according to required format, using None to indicate failure
    print("FinalAnswer=【None】")