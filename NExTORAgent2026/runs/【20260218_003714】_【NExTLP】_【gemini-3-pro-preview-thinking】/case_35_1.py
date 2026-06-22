import gurobipy as gp
from gurobipy import GRB

# 2. Define all parameter matrices and data inputs
num_routes = 12
nodes = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨']
num_streets = 17
travel_time = 25
rest_time = 5
round_trip_time = 60

# Required passing intervals for each street (min)
intervals = {
    '①②': 10, '①④': 10, '①⑦': 15, '②③': 10, '②④': 10,
    '③④': 15, '③⑤': 15, '③⑥': 15, '④⑤': 10, '④⑦': 15,
    '④⑧': 10, '⑤⑥': 10, '⑤⑧': 15, '⑥⑧': 20, '⑦⑧': 20,
    '⑦⑨': 20, '⑧⑨': 20
}

# Route definitions (index 0 is None to align 1-based indexing)
routes_list = [
    None,
    ['①', '②', '③', '⑥'],
    ['①', '④', '⑤', '⑥'],
    ['①', '④', '⑧', '⑨'],
    ['①', '②', '④', '⑦', '⑨'],
    ['①', '②', '③', '⑤', '⑧'],
    ['①', '⑦', '⑨', '⑧', '⑥'],
    ['③', '④', '⑧', '⑥'],
    ['②', '④', '⑦', '⑧'],
    ['⑦', '④', '③', '⑥'],
    ['②', '③', '⑤', '⑥'],
    ['③', '④', '⑤', '⑧', '⑨'],
    ['①', '⑦', '⑧', '⑤', '⑥']
]

# Create Gurobi Model
model = gp.Model("Bus_Route_Optimization")

# 3. Create decision variables
# x[i] represents the number of buses assigned to route i
x = model.addVars(range(1, num_routes + 1), vtype=GRB.INTEGER, lb=0, name="x")

# 5. Set up the objective function
# Minimize total number of vehicles
model.setObjective(gp.quicksum(x[i] for i in range(1, num_routes + 1)), GRB.MINIMIZE)

# 6. Add all constraints
# For each street segment, ensure the sum of buses on routes traversing it meets the frequency requirement.
# Frequency required (buses/hr) = 60 / interval (min)
for segment, interval in intervals.items():
    # Parse the nodes defining the street
    u, v = segment[0], segment[1]
    
    # Calculate required number of buses
    # Note: 60/interval is exact integer for all given values (10, 15, 20)
    req_buses = round_trip_time / interval
    
    # Identify routes that cover this street segment
    covering_routes = []
    for r_idx in range(1, num_routes + 1):
        path = routes_list[r_idx]
        # Check if u and v are adjacent in the path
        for k in range(len(path) - 1):
            if (path[k] == u and path[k+1] == v) or (path[k] == v and path[k+1] == u):
                covering_routes.append(x[r_idx])
                break
    
    # Add constraint: sum(x_r) >= required_buses
    model.addConstr(gp.quicksum(covering_routes) >= req_buses, name=f"Interval_{segment}")

# 7. Solve the model and print results
model.optimize()

if model.status == GRB.OPTIMAL:
    print("\nOptimal Solution Found:")
    total_vehicles = int(model.objVal)
    print(f"Total number of vehicles: {total_vehicles}")
    for i in range(1, num_routes + 1):
        val = int(x[i].X)
        if val > 0:
            print(f"Route {i}: {val} buses")
            
    print(f"FinalAnswer=【{total_vehicles}】")
else:
    print("No optimal solution found.")