import gurobipy as gp
from gurobipy import GRB

# Define the model
model = gp.Model("BusRouteOptimization")

# Parameter matrices and data inputs
num_routes = 12
nodes = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨']
num_streets = 17
travel_time = 25
rest_time = 5
round_trip_time = 60
# Street names and their required bus intervals (min)
intervals = {
    '①②': 10, '①④': 10, '①⑦': 15, '②③': 10, '②④': 10, 
    '③④': 15, '③⑤': 15, '③⑥': 15, '④⑤': 10, '④⑦': 15, 
    '④⑧': 10, '⑤⑥': 10, '⑤⑧': 15, '⑥⑧': 20, '⑦⑧': 20, 
    '⑦⑨': 20, '⑧⑨': 20
}

# Calculated frequency requirements (buses per hour)
# Frequency = 60 / Interval
f_req = {street: round_trip_time // interval for street, interval in intervals.items()}

# Create decision variables
# x[i] is the number of buses assigned to route i (i=1 to 12)
x = model.addVars(range(1, 13), vtype=GRB.INTEGER, lb=0, name="x")

# Set up the objective function: Minimize the total number of vehicles
model.setObjective(gp.quicksum(x[i] for i in range(1, 13)), GRB.MINIMIZE)

# Add all constraints based on street coverage
# Each constraint ensures the total number of buses from all routes passing through a street 
# meets the required density (frequency per hour).

model.addConstr(x[1] + x[4] + x[5] >= f_req['①②'], "Interval_1_2")
model.addConstr(x[2] + x[3] >= f_req['①④'], "Interval_1_4")
model.addConstr(x[6] + x[12] >= f_req['①⑦'], "Interval_1_7")
model.addConstr(x[1] + x[5] + x[10] >= f_req['②③'], "Interval_2_3")
model.addConstr(x[4] + x[8] >= f_req['②④'], "Interval_2_4")
model.addConstr(x[7] + x[9] + x[11] >= f_req['③④'], "Interval_3_4")
model.addConstr(x[5] + x[10] >= f_req['③⑤'], "Interval_3_5")
model.addConstr(x[1] + x[9] >= f_req['③⑥'], "Interval_3_6")
model.addConstr(x[2] + x[11] >= f_req['④⑤'], "Interval_4_5")
model.addConstr(x[4] + x[8] + x[9] >= f_req['④⑦'], "Interval_4_7")
model.addConstr(x[3] + x[7] >= f_req['④⑧'], "Interval_4_8")
model.addConstr(x[2] + x[10] + x[12] >= f_req['⑤⑥'], "Interval_5_6")
model.addConstr(x[5] + x[11] + x[12] >= f_req['⑤⑧'], "Interval_5_8")
model.addConstr(x[6] + x[7] >= f_req['⑥⑧'], "Interval_6_8")
model.addConstr(x[8] + x[12] >= f_req['⑦⑧'], "Interval_7_8")
model.addConstr(x[4] + x[6] >= f_req['⑦⑨'], "Interval_7_9")
model.addConstr(x[3] + x[6] + x[11] >= f_req['⑧⑨'], "Interval_8_9")

# Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    total_vehicles = int(model.objVal)
    print(f"Optimal total number of vehicles: {total_vehicles}")
    for i in range(1, 13):
        print(f"Route {i}: {int(x[i].X)} buses")
    print(f"FinalAnswer=【{total_vehicles}】")
else:
    print("No optimal solution found.")