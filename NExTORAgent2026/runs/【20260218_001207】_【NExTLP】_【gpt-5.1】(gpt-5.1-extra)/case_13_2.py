import gurobipy as gp

# Huifeng Airlines fleet scheduling MILP with Gurobi
# Minimize operating + sanitation cost, subject to fleet, demand, and policy constraints

# 1. Create model
model = gp.Model("Huifeng_Airlines_Fleet_Scheduling")

# 2. Define all parameter matrices and data inputs (from Parameters List)

# Basic parameters
num_hub_cities = 5
num_aircraft_type_A = 8
num_aircraft_type_B = 9
num_aircraft_type_C = 10
flight_indices = [1, 2, 3, 4, 5]
max_flights_per_aircraft = 1
max_same_model_per_flight = 3
must_use_large_aircraft_flight = [5]
large_aircraft_models = ['A', 'B']
forbidden_flight_model_pairs = [{'flight': 4, 'model': 'B'}]
passenger_capacity_per_aircraft = [None, None, None]  # unused directly, but kept as given

# Capacity per model
passenger_capacity_model = [
    {'model': 'A', 'capacity': 200},
    {'model': 'B', 'capacity': 180},
    {'model': 'C', 'capacity': 100},
]

# Sanitation cost per passenger per model
sanitation_cost_per_person_model = [
    {'model': 'A', 'cost': 0.1},
    {'model': 'B', 'cost': 0.1},
    {'model': 'C', 'cost': 0.2},
]

# Operating costs table (ten-thousand yuan per aircraft-flight)
Table_1_Costs = {
    'flights': [1, 2, 3, 4, 5],
    'Model_A_Cost': [10, 9, 7, 8, 11],
    'Model_B_Cost': [8, 8, 6, 7, 12],
    'Model_C_Cost': [8, 5, 4, 7, 3],
    'unit': 'ten_thousand_yuan',
}

# Demand table (passengers)
Table_2_Demand = {
    'flights': [1, 2, 3, 4, 5],
    'demand': [500, 1000, 330, 290, 470],
}

# Build lookup dictionaries
cap = {p['model']: p['capacity'] for p in passenger_capacity_model}
san_cost = {p['model']: p['cost'] for p in sanitation_cost_per_person_model}

cost_A = {
    f: c for f, c in zip(Table_1_Costs['flights'], Table_1_Costs['Model_A_Cost'])
}
cost_B = {
    f: c for f, c in zip(Table_1_Costs['flights'], Table_1_Costs['Model_B_Cost'])
}
cost_C = {
    f: c for f, c in zip(Table_1_Costs['flights'], Table_1_Costs['Model_C_Cost'])
}
demand = {
    f: d for f, d in zip(Table_2_Demand['flights'], Table_2_Demand['demand'])
}

# 3. Create decision variables
# x_A[i], x_B[i], x_C[i] ∈ {0,1,2,3}  (integer number of aircraft of each type on flight i)
x_A = model.addVars(
    flight_indices,
    vtype=gp.GRB.INTEGER,
    lb=0,
    ub=max_same_model_per_flight,
    name="x_A",
)
x_B = model.addVars(
    flight_indices,
    vtype=gp.GRB.INTEGER,
    lb=0,
    ub=max_same_model_per_flight,
    name="x_B",
)
x_C = model.addVars(
    flight_indices,
    vtype=gp.GRB.INTEGER,
    lb=0,
    ub=max_same_model_per_flight,
    name="x_C",
)

# Apply special policy constraints that fix some variables to 0
# Flight 5 must use large aircraft (A,B) -> no C on flight 5
for f in must_use_large_aircraft_flight:
    # explicitly enforce x_C[f] = 0 with bounds
    x_C[f].lb = 0
    x_C[f].ub = 0

# Type B forbidden on flight 4
for pair in forbidden_flight_model_pairs:
    if pair['model'] == 'B':
        f = pair['flight']
        x_B[f].lb = 0
        x_B[f].ub = 0

# 5. Set up the objective function
# Minimize:
#   Σ_f (c_Af x_Af + c_Bf x_Bf + c_Cf x_Cf)
# + Σ_f [0.1(200 x_Af + 180 x_Bf) + 0.2(100 x_Cf)]

expr_cost = gp.quicksum(
    cost_A[f] * x_A[f] +
    cost_B[f] * x_B[f] +
    cost_C[f] * x_C[f]
    for f in flight_indices
)

expr_sanitation = gp.quicksum(
    san_cost['A'] * cap['A'] * x_A[f] +
    san_cost['B'] * cap['B'] * x_B[f] +
    san_cost['C'] * cap['C'] * x_C[f]
    for f in flight_indices
)

model.setObjective(expr_cost + expr_sanitation, gp.GRB.MINIMIZE)

# 6. Add all constraints

# 6.1 Fleet availability constraints
model.addConstr(
    gp.quicksum(x_A[f] for f in flight_indices) <= num_aircraft_type_A,
    name="Type_A_fleet",
)
model.addConstr(
    gp.quicksum(x_B[f] for f in flight_indices) <= num_aircraft_type_B,
    name="Type_B_fleet",
)
model.addConstr(
    gp.quicksum(x_C[f] for f in flight_indices) <= num_aircraft_type_C,
    name="Type_C_fleet",
)

# 6.2 Demand satisfaction constraints
for f in flight_indices:
    model.addConstr(
        cap['A'] * x_A[f] + cap['B'] * x_B[f] + cap['C'] * x_C[f] >= demand[f],
        name=f"Demand_f{f}",
    )

# Note: per-flight per-model upper bounds (≤ 3) are already enforced via variable upper bounds

# There are no indicator constraints required by the problem description,
# so we do not use addGenConstrIndicator here.

# 7. Solve the model and print results

model.optimize()

if model.status == gp.GRB.OPTIMAL:
    # Print detailed solution
    print("Optimal objective value (total cost in 10,000 yuan units + sanitation cost):", model.objVal)
    print("Assignment of aircraft per flight:")
    for f in flight_indices:
        print(
            f"Flight {f}: "
            f"A = {int(round(x_A[f].X))}, "
            f"B = {int(round(x_B[f].X))}, "
            f"C = {int(round(x_C[f].X))}"
        )
    # FinalAnswer must be the numeric answer to the question: minimum total cost
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found. Model status:", model.status)
    # In case of no optimal solution, we can still output a placeholder
    print("FinalAnswer=【None】")