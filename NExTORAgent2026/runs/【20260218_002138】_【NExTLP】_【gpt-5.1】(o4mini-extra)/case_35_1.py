import gurobipy as gp
from gurobipy import GRB

# ==============================
# 1. Parameters (from Parameters List)
# ==============================
num_routes = 12
nodes = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨']
num_streets = 17
travel_time = 25
rest_time = 5
round_trip_time = 60
intervals = {
    '①②': 10,
    '①④': 10,
    '①⑦': 15,
    '②③': 10,
    '②④': 10,
    '③④': 15,
    '③⑤': 15,
    '③⑥': 15,
    '④⑤': 10,
    '④⑦': 15,
    '④⑧': 10,
    '⑤⑥': 10,
    '⑤⑧': 15,
    '⑥⑧': 20,
    '⑦⑧': 20,
    '⑦⑨': 20,
    '⑧⑨': 20
}
routes = [
    None,  # index 0 is dummy to keep 1-based indexing
    ['①', '②', '③', '⑥'],              # 1
    ['①', '④', '⑤', '⑥'],              # 2
    ['①', '④', '⑧', '⑨'],              # 3
    ['①', '②', '④', '⑦', '⑨'],         # 4
    ['①', '②', '③', '⑤', '⑧'],         # 5
    ['①', '⑦', '⑨', '⑧', '⑥'],         # 6
    ['③', '④', '⑧', '⑥'],              # 7
    ['②', '④', '⑦', '⑧'],              # 8
    ['⑦', '④', '③', '⑥'],              # 9
    ['②', '③', '⑤', '⑥'],              # 10
    ['③', '④', '⑤', '⑧', '⑨'],         # 11
    ['①', '⑦', '⑧', '⑤', '⑥']          # 12
]

# ==============================
# 2. Create model
# ==============================
model = gp.Model("DongfangCity_BusRoute_Optimization")

# ==============================
# 3. Decision Variables
# ==============================
# x[i] = number of buses assigned to route i
x = model.addVars(range(1, num_routes + 1),
                  vtype=GRB.INTEGER,
                  lb=0,
                  name="x")

# ==============================
# 4. Objective: minimize total number of buses
# ==============================
model.setObjective(gp.quicksum(x[i] for i in range(1, num_routes + 1)), GRB.MINIMIZE)

# ==============================
# 5. Constraints (interval coverage on each street)
#    All constraints are taken directly from the validated model.
# ==============================

# ①② : x1 + x4 + x5 ≥ 6
model.addConstr(x[1] + x[4] + x[5] >= 6, name="Interval_1_2")

# ①④ : x2 + x3 ≥ 6
model.addConstr(x[2] + x[3] >= 6, name="Interval_1_4")

# ①⑦ : x6 + x12 ≥ 4
model.addConstr(x[6] + x[12] >= 4, name="Interval_1_7")

# ②③ : x1 + x5 + x10 ≥ 6
model.addConstr(x[1] + x[5] + x[10] >= 6, name="Interval_2_3")

# ②④ : x4 + x8 ≥ 6
model.addConstr(x[4] + x[8] >= 6, name="Interval_2_4")

# ③④ : x7 + x9 + x11 ≥ 4
model.addConstr(x[7] + x[9] + x[11] >= 4, name="Interval_3_4")

# ③⑤ : x5 + x10 ≥ 4
model.addConstr(x[5] + x[10] >= 4, name="Interval_3_5")

# ③⑥ : x1 + x9 ≥ 4
model.addConstr(x[1] + x[9] >= 4, name="Interval_3_6")

# ④⑤ : x2 + x11 ≥ 6
model.addConstr(x[2] + x[11] >= 6, name="Interval_4_5")

# ④⑦ : x4 + x8 + x9 ≥ 4
model.addConstr(x[4] + x[8] + x[9] >= 4, name="Interval_4_7")

# ④⑧ : x3 + x7 ≥ 6
model.addConstr(x[3] + x[7] >= 6, name="Interval_4_8")

# ⑤⑥ : x2 + x10 + x12 ≥ 6
model.addConstr(x[2] + x[10] + x[12] >= 6, name="Interval_5_6")

# ⑤⑧ : x5 + x11 + x12 ≥ 4
model.addConstr(x[5] + x[11] + x[12] >= 4, name="Interval_5_8")

# ⑥⑧ : x6 + x7 ≥ 3
model.addConstr(x[6] + x[7] >= 3, name="Interval_6_8")

# ⑦⑧ : x8 + x12 ≥ 3
model.addConstr(x[8] + x[12] >= 3, name="Interval_7_8")

# ⑦⑨ : x4 + x6 ≥ 3
model.addConstr(x[4] + x[6] >= 3, name="Interval_7_9")

# ⑧⑨ : x3 + x6 + x11 ≥ 3
model.addConstr(x[3] + x[6] + x[11] >= 3, name="Interval_8_9")

# ==============================
# 6. Optimize model
# ==============================
model.optimize()

# ==============================
# 7. Print solution
# ==============================
if model.status == GRB.OPTIMAL:
    print("Optimal solution found.")
    total_buses = 0
    for i in range(1, num_routes + 1):
        val = int(round(x[i].X))
        print(f"Route {i}: x_{i} = {val}")
        total_buses += val

    print(f"Total number of buses (objective value): {total_buses}")

    # Final required answer:
    print(f"FinalAnswer=【{total_buses}】")
else:
    print("No optimal solution found.")
    # In case of infeasibility or other status, still output something, per required format.
    print("FinalAnswer=【None】")