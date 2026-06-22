import gurobipy as gp
from gurobipy import GRB

# Define parameters from the Parameters List
aircraft_supply = {'A': 8, 'B': 9, 'C': 10}
capacity = {'A': 200, 'B': 180, 'C': 100}
sanitation_cost = {'A': 0.1, 'B': 0.1, 'C': 0.2}
flight_cost = {
    'A': [10, 9, 7, 8, 11],
    'B': [8, 8, 6, 7, 12],
    'C': [8, 5, 4, 7, 3]
}
demand = [500, 1000, 330, 290, 470]
num_flights = 5
max_per_flight = 3

# Create model
model = gp.Model("FlightScheduling")

# Decision variables: x[i][k] - number of aircraft of type k assigned to flight i
x = {}
for i in range(num_flights):
    for k in ['A', 'B', 'C']:
        x[i,k] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_per_flight, 
                               name=f"x_{i+1}_{k}")

# Set objective: minimize total cost (operation cost + sanitation cost)
obj_expr = gp.QuadExpr()
for i in range(num_flights):
    for k in ['A', 'B', 'C']:
        # Operation cost (from table, in 10,000 yuan)
        op_cost = flight_cost[k][i]
        # Sanitation cost per aircraft = sanitation_cost * capacity
        sanit_per_ac = sanitation_cost[k] * capacity[k]
        # Total cost coefficient = operation cost + sanitation cost
        total_coeff = op_cost + sanit_per_ac
        obj_expr += total_coeff * x[i,k]

model.setObjective(obj_expr, GRB.MINIMIZE)

# Constraints: Fleet availability
for k, supply in aircraft_supply.items():
    model.addConstr(gp.quicksum(x[i,k] for i in range(num_flights)) <= supply, 
                    name=f"supply_{k}")

# Constraints: Demand satisfaction
for i in range(num_flights):
    demand_expr = gp.QuadExpr()
    for k in ['A', 'B', 'C']:
        demand_expr += capacity[k] * x[i,k]
    model.addConstr(demand_expr >= demand[i], name=f"demand_{i+1}")

# Constraints: Flight 5 must use larger aircraft (no type C)
model.addConstr(x[4,'C'] == 0, name="flight5_no_C")

# Constraints: Type B cannot perform flight 4
model.addConstr(x[3,'B'] == 0, name="flight4_no_B")

# Solve the model
model.optimize()

# Check if optimal solution is found
if model.status == GRB.OPTIMAL:
    # Calculate total cost
    total_cost = model.objVal
    print(f"Optimal total cost: {total_cost:.4f} (in 10,000 yuan)")
    
    # Print detailed assignment
    print("\nAircraft assignment:")
    for i in range(num_flights):
        print(f"Flight {i+1} (demand {demand[i]}): ", end="")
        for k in ['A', 'B', 'C']:
            if x[i,k].x > 0:
                print(f"{k}: {int(x[i,k].x)} ", end="")
        print()
    
    print(f"FinalAnswer=【{total_cost:.4f}】")
else:
    print("No optimal solution found")
    print(f"FinalAnswer=【0】")