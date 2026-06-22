import gurobipy as gp
from gurobipy import GRB

# 1. Define parameters and data inputs
aircraft_supply = {'A': 8, 'B': 9, 'C': 10}
number_of_flights = 5
flights = [1, 2, 3, 4, 5]
max_same_model_per_flight = 3
flights_require_large = [5]  # Flight 5 cannot use small aircraft (Type C)
forbidden_B_flights = [4]    # Flight 4 cannot use Type B
capacity = {'A': 200, 'B': 180, 'C': 100}
sanitation_cost_per_person = {'A': 0.1, 'B': 0.1, 'C': 0.2}

# Operating costs (10,000 yuan) from Table 1
# Dictionary mapping aircraft type to list of costs for flights 1 to 5
operating_costs = {
    'A': [10, 9, 7, 8, 11],
    'B': [8, 8, 6, 7, 12],
    'C': [8, 5, 4, 7, 3]
}

# Demand (number of passengers) for flights 1 to 5
flight_demand = [500, 1000, 330, 290, 470]

# Calculate total cost coefficients: Operating Cost + (Sanitation Cost/person * Capacity)
# Total cost unit will be consistent with operating cost unit (10,000 yuan is implied base, 
# but sanitation calculation seems absolute. Assuming mixing directly as per objective function formula provided in problem context).
# Context formula: (c_{i,A} + 0.1*200) x... where c_{i,A} is from table (e.g., 10).
# So we simply add them.
cost_coefficients = {}
for k in ['A', 'B', 'C']:
    sanitation_total = sanitation_cost_per_person[k] * capacity[k]
    for i_idx, f in enumerate(flights):
        # operating_costs list is 0-indexed
        op_cost = operating_costs[k][i_idx]
        cost_coefficients[(f, k)] = op_cost + sanitation_total

# 2. Create the model
model = gp.Model("Huifeng_Airlines_Scheduling")

# 3. Create decision variables
# x[i, k] = number of type k aircraft assigned to flight i
x = model.addVars(flights, ['A', 'B', 'C'], vtype=GRB.INTEGER, lb=0, name="x")

# 4. Set up the objective function
# Minimize total cost
model.setObjective(
    gp.quicksum(cost_coefficients[(i, k)] * x[i, k] for i in flights for k in ['A', 'B', 'C']),
    GRB.MINIMIZE
)

# 5. Add constraints

# (1) Demand satisfaction for each flight
for i_idx, i in enumerate(flights):
    model.addConstr(
        gp.quicksum(capacity[k] * x[i, k] for k in ['A', 'B', 'C']) >= flight_demand[i_idx],
        name=f"Demand_Flight_{i}"
    )

# (2) Fleet supply limits
for k in ['A', 'B', 'C']:
    model.addConstr(
        gp.quicksum(x[i, k] for i in flights) <= aircraft_supply[k],
        name=f"Supply_Limit_{k}"
    )

# (3) Maximum same model per flight limit
for i in flights:
    for k in ['A', 'B', 'C']:
        model.addConstr(x[i, k] <= max_same_model_per_flight, name=f"Max_3_{i}_{k}")

# (4) Specific flight restrictions
# Flight 5 must be operated by larger aircraft (No Type C)
for f in flights_require_large:
    model.addConstr(x[f, 'C'] == 0, name=f"No_C_Flight_{f}")

# Flight 4 cannot be operated by Type B
for f in forbidden_B_flights:
    model.addConstr(x[f, 'B'] == 0, name=f"No_B_Flight_{f}")

# 6. Solve the model
model.optimize()

# 7. Print results
if model.status == GRB.OPTIMAL:
    print("Optimal Solution Found:")
    total_cost = model.ObjVal
    for i in flights:
        print(f"Flight {i} Assignments:")
        for k in ['A', 'B', 'C']:
            val = x[i, k].X
            if val > 0.5:
                print(f"  Type {k}: {int(val)}")
    
    # Output the final answer in the requested format
    print(f"FinalAnswer=【{total_cost}】")
else:
    print("No optimal solution found.")