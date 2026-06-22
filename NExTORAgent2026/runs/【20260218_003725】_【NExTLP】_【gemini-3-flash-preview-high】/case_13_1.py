import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs
aircraft_supply = {'A': 8, 'B': 9, 'C': 10}
number_of_flights = 5
max_same_model_per_flight = 3
flights_require_large = [5]  # Index starting from 1
forbidden_B_flights = [4]    # Index starting from 1
capacity = {'A': 200, 'B': 180, 'C': 100}
sanitation_cost = {'A': 0.1, 'B': 0.1, 'C': 0.2}

Table_1_Model_Costs = {
    'A': [10, 9, 7, 8, 11],
    'B': [8, 8, 6, 7, 12],
    'C': [8, 5, 4, 7, 3]
}

Table_2_Flight_Demand = [500, 1000, 330, 290, 470]

# Pre-calculate cost coefficients (including operational cost and sanitation cost)
# sanitation_cost * capacity is added to the flight-specific operational cost
# Note: indices in Python are 0..4, representing flights 1..5
aircraft_types = ['A', 'B', 'C']
flights = range(number_of_flights)

cost_coeffs = {}
for k in aircraft_types:
    for i in flights:
        cost_coeffs[i, k] = Table_1_Model_Costs[k][i] + sanitation_cost[k] * capacity[k]

# 2. Create the Gurobi model
model = gp.Model("Huifeng_Airlines_Scheduling")

# 3. Create decision variables
# x[i, k] is the number of aircraft of type k used for flight i+1
x = model.addVars(flights, aircraft_types, vtype=GRB.INTEGER, lb=0, ub=max_same_model_per_flight, name="x")

# 4. Set up the objective function
model.setObjective(
    gp.quicksum(cost_coeffs[i, k] * x[i, k] for i in flights for k in aircraft_types),
    GRB.MINIMIZE
)

# 5. Add constraints

# Aircraft Supply Constraints: Total aircraft used per type across all flights cannot exceed supply
for k in aircraft_types:
    model.addConstr(gp.quicksum(x[i, k] for i in flights) <= aircraft_supply[k], name=f"Supply_{k}")

# Flight Capacity (Demand) Constraints: Total capacity on each flight must meet demand
for i in flights:
    model.addConstr(
        gp.quicksum(capacity[k] * x[i, k] for k in aircraft_types) >= Table_2_Flight_Demand[i],
        name=f"Demand_Flight_{i+1}"
    )

# Specific Flight Restrictions
# Flight 5 (index 4) must be operated by a larger aircraft (No type C aircraft)
for f_idx in flights_require_large:
    model.addConstr(x[f_idx - 1, 'C'] == 0, name=f"Flight_{f_idx}_NoC")

# Model B aircraft cannot perform flight 4 (index 3)
for f_idx in forbidden_B_flights:
    model.addConstr(x[f_idx - 1, 'B'] == 0, name=f"Flight_{f_idx}_NoB")

# 6. Solve the model
model.optimize()

# 7. Print results
if model.status == GRB.OPTIMAL:
    print(f"Optimal Objective Value: {model.ObjVal}")
    for i in flights:
        for k in aircraft_types:
            if x[i, k].X > 0:
                print(f"Flight {i+1}, Aircraft Type {k}: {int(x[i, k].X)}")
    print(f"FinalAnswer=【{int(model.ObjVal)}】")
else:
    print("No optimal solution found.")