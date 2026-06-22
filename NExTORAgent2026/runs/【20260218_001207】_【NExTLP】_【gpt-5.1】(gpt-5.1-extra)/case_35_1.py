import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters (given)
# =========================
bus_commute_share_min = 45.0
num_original_routes = 12
num_nodes = 9
num_main_streets = 17

Table_1_RequiredInterval = [
    ['①②', 10],
    ['①④', 10],
    ['①⑦', 15],
    ['②③', 10],
    ['②④', 10],
    ['③④', 15],
    ['③⑤', 15],
    ['③⑥', 15],
    ['④⑤', 10],
    ['④⑦', 15],
    ['④⑧', 10],
    ['⑤⑥', 10],
    ['⑤⑧', 15],
    ['⑥⑧', 20],
    ['⑦⑧', 20],
    ['⑦⑨', 20],
    ['⑧⑨', 20]
]

Table_2_Routes = [
    [1, ['①', '②', '③', '⑥']],
    [2, ['①', '④', '⑤', '⑥']],
    [3, ['①', '④', '⑧', '⑨']],
    [4, ['①', '②', '④', '⑦', '⑨']],
    [5, ['①', '②', '③', '⑤', '⑧']],
    [6, ['①', '⑦', '⑨', '⑧', '⑥']],
    [7, ['③', '④', '⑧', '⑥']],
    [8, ['②', '④', '⑦', '⑧']],
    [9, ['⑦', '④', '③', '⑥']],
    [10, ['②', '③', '⑤', '⑥']],
    [11, ['③', '④', '⑤', '⑧', '⑨']],
    [12, ['①', '⑦', '⑧', '⑤', '⑥']]
]

Street_Nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
Street_Edges = [
    [1, 2],
    [1, 4],
    [1, 7],
    [2, 3],
    [2, 4],
    [3, 5],
    [3, 6],
    [4, 3],
    [4, 5],
    [4, 8],
    [5, 6],
    [7, 4],
    [7, 8],
    [7, 9],
    [8, 5],
    [8, 6],
    [9, 8]
]

route_travel_time_one_way_min = 25
terminal_rest_time_min = 5
route_round_trip_cycle_time_min = 60
route_round_trips_per_hour = 1

# =========================
# 2. Create model
# =========================
model = gp.Model("Dongfang_City_Bus_Line_Optimization")

# =========================
# 3. Decision variables
# =========================
x = {}
for i in range(1, num_original_routes + 1):
    x[i] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x{i}")

model.update()

# =========================
# 4. Objective function
# =========================
model.setObjective(
    gp.quicksum(x[i] for i in range(1, num_original_routes + 1)),
    GRB.MINIMIZE
)

# =========================
# 5. Constraints
# =========================

# Street 1-2: x1 + x4 + x5 >= 6
model.addConstr(x[1] + x[4] + x[5] >= 6, name="Street_1_2")

# Street 1-4: x2 + x3 >= 6
model.addConstr(x[2] + x[3] >= 6, name="Street_1_4")

# Street 1-7: x6 + x12 >= 4
model.addConstr(x[6] + x[12] >= 4, name="Street_1_7")

# Street 2-3: x1 + x5 + x10 >= 6
model.addConstr(x[1] + x[5] + x[10] >= 6, name="Street_2_3")

# Street 2-4: x2 + x4 + x8 >= 6
model.addConstr(x[2] + x[4] + x[8] >= 6, name="Street_2_4")

# Street 3-4: x7 + x9 + x11 >= 4
model.addConstr(x[7] + x[9] + x[11] >= 4, name="Street_3_4")

# Street 3-5: x5 + x10 + x11 >= 4
model.addConstr(x[5] + x[10] + x[11] >= 4, name="Street_3_5")

# Street 3-6: x1 + x9 >= 4
model.addConstr(x[1] + x[9] >= 4, name="Street_3_6")

# Street 4-5: x2 + x11 >= 6
model.addConstr(x[2] + x[11] >= 6, name="Street_4_5")

# Street 4-7: x4 + x8 + x9 >= 4
model.addConstr(x[4] + x[8] + x[9] >= 4, name="Street_4_7")

# Street 4-8: x3 + x7 >= 6
model.addConstr(x[3] + x[7] >= 6, name="Street_4_8")

# Street 5-6: x2 + x10 + x12 >= 6
model.addConstr(x[2] + x[10] + x[12] >= 6, name="Street_5_6")

# Street 5-8: x5 + x11 + x12 >= 4
model.addConstr(x[5] + x[11] + x[12] >= 4, name="Street_5_8")

# Street 6-8: x6 + x7 >= 3
model.addConstr(x[6] + x[7] >= 3, name="Street_6_8")

# Street 7-8: x8 + x12 >= 3
model.addConstr(x[8] + x[12] >= 3, name="Street_7_8")

# Street 7-9: x4 + x6 >= 3
model.addConstr(x[4] + x[6] >= 3, name="Street_7_9")

# Street 8-9: x3 + x6 + x11 >= 3
model.addConstr(x[3] + x[6] + x[11] >= 3, name="Street_8_9")

# =========================
# 6. Optimize
# =========================
model.optimize()

# =========================
# 7. Print results
# =========================
if model.Status == GRB.OPTIMAL:
    print("Optimal solution found.")
    total_buses = 0
    for i in range(1, num_original_routes + 1):
        val = int(round(x[i].X))
        print(f"x{i} (buses on route {i}) = {val}")
        total_buses += val
    print(f"Total number of buses = {total_buses}")
    # FinalAnswer is the total number of buses
    print(f"FinalAnswer=【{total_buses}】")
else:
    print("No optimal solution found.")
    # If no solution, still output a FinalAnswer (e.g., -1 to indicate infeasibility or no opt)
    print("FinalAnswer=【-1】")