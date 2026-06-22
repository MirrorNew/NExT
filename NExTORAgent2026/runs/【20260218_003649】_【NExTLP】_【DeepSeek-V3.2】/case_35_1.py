import gurobipy as gp

# Parameters from the Parameters List
num_routes = 12
intervals = {'①②': 10, '①④': 10, '①⑦': 15, '②③': 10, '②④': 10, '③④': 15, '③⑤': 15, '③⑥': 15, '④⑤': 10, '④⑦': 15, '④⑧': 10, '⑤⑥': 10, '⑤⑧': 15, '⑥⑧': 20, '⑦⑧': 20, '⑦⑨': 20, '⑧⑨': 20}
routes = [None, ['①', '②', '③', '⑥'], ['①', '④', '⑤', '⑥'], ['①', '④', '⑧', '⑨'], ['①', '②', '④', '⑦', '⑨'], ['①', '②', '③', '⑤', '⑧'], ['①', '⑦', '⑨', '⑧', '⑥'], ['③', '④', '⑧', '⑥'], ['②', '④', '⑦', '⑧'], ['⑦', '④', '③', '⑥'], ['②', '③', '⑤', '⑥'], ['③', '④', '⑤', '⑧', '⑨'], ['①', '⑦', '⑧', '⑤', '⑥']]

# Create model
model = gp.Model("BusRouteOptimization")

# Decision variables: number of buses for each route
x = {}
for i in range(1, num_routes + 1):
    x[i] = model.addVar(vtype=gp.GRB.INTEGER, lb=0, name=f"x_{i}")

# Set objective: minimize total number of buses
model.setObjective(gp.quicksum(x[i] for i in range(1, num_routes + 1)), gp.GRB.MINIMIZE)

# Add constraints for each street based on required bus intervals
# Street ①②: x1 + x4 + x5 >= 6
model.addConstr(x[1] + x[4] + x[5] >= 6, "Street_12")

# Street ①④: x2 + x3 >= 6
model.addConstr(x[2] + x[3] >= 6, "Street_14")

# Street ①⑦: x6 + x12 >= 4
model.addConstr(x[6] + x[12] >= 4, "Street_17")

# Street ②③: x1 + x5 + x10 >= 6
model.addConstr(x[1] + x[5] + x[10] >= 6, "Street_23")

# Street ②④: x4 + x8 >= 6
model.addConstr(x[4] + x[8] >= 6, "Street_24")

# Street ③④: x7 + x9 + x11 >= 4
model.addConstr(x[7] + x[9] + x[11] >= 4, "Street_34")

# Street ③⑤: x5 + x10 >= 4
model.addConstr(x[5] + x[10] >= 4, "Street_35")

# Street ③⑥: x1 + x9 >= 4
model.addConstr(x[1] + x[9] >= 4, "Street_36")

# Street ④⑤: x2 + x11 >= 6
model.addConstr(x[2] + x[11] >= 6, "Street_45")

# Street ④⑦: x4 + x8 + x9 >= 4
model.addConstr(x[4] + x[8] + x[9] >= 4, "Street_47")

# Street ④⑧: x3 + x7 >= 6
model.addConstr(x[3] + x[7] >= 6, "Street_48")

# Street ⑤⑥: x2 + x10 + x12 >= 6
model.addConstr(x[2] + x[10] + x[12] >= 6, "Street_56")

# Street ⑤⑧: x5 + x11 + x12 >= 4
model.addConstr(x[5] + x[11] + x[12] >= 4, "Street_58")

# Street ⑥⑧: x6 + x7 >= 3
model.addConstr(x[6] + x[7] >= 3, "Street_68")

# Street ⑦⑧: x8 + x12 >= 3
model.addConstr(x[8] + x[12] >= 3, "Street_78")

# Street ⑦⑨: x4 + x6 >= 3
model.addConstr(x[4] + x[6] >= 3, "Street_79")

# Street ⑧⑨: x3 + x6 + x11 >= 3
model.addConstr(x[3] + x[6] + x[11] >= 3, "Street_89")

# Solve the model
model.optimize()

# Check if optimal solution found
if model.status == gp.GRB.OPTIMAL:
    total_vehicles = int(model.ObjVal)
    print(f"Optimal total number of vehicles: {total_vehicles}")
    print("\nBus allocation per route:")
    for i in range(1, num_routes + 1):
        print(f"Route {i}: {int(x[i].X)} buses")
else:
    print("No optimal solution found")
    total_vehicles = None

print(f"FinalAnswer=【{total_vehicles}】")