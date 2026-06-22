import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters (given)
# =========================

aircraft_supply = {'A': 8, 'B': 9, 'C': 10}
number_of_flights = 5
max_same_model_per_flight = 3
flights_require_large = [5]          # Flight(s) that cannot use small C type (enforced below)
forbidden_B_flights = [4]            # Flight(s) that cannot use B type
capacity = {'A': 200, 'B': 180, 'C': 100}
sanitation_cost = {'A': 0.1, 'B': 0.1, 'C': 0.2}
Table_1_Model_Costs = {
    'A': [10, 9, 7, 8, 11],
    'B': [8, 8, 6, 7, 12],
    'C': [8, 5, 4, 7, 3]
}
Table_2_Flight_Demand = [500, 1000, 330, 290, 470]

# Effective per-aircraft cost = operating cost + sanitation cost * capacity
# All monetary units kept as given (10,000 yuan)
effective_cost = {k: [] for k in ['A', 'B', 'C']}
for k in ['A', 'B', 'C']:
    for i in range(number_of_flights):
        eff = Table_1_Model_Costs[k][i] + sanitation_cost[k] * capacity[k]
        effective_cost[k].append(eff)

# =========================
# 2. Model
# =========================

model = gp.Model("Huifeng_Airlines_Fleet_Scheduling")

flights = range(number_of_flights)
types = ['A', 'B', 'C']

# =========================
# 3. Decision variables
# =========================
# x[i,k] = number of aircraft of type k assigned to flight i (integer, >=0)
x = model.addVars(
    flights,
    types,
    vtype=GRB.INTEGER,
    lb=0,
    name="x"
)

# =========================
# 4. Objective function
# =========================

model.setObjective(
    gp.quicksum(
        effective_cost[k][i] * x[i, k] for i in flights for k in types
    ),
    GRB.MINIMIZE
)

# =========================
# 5. Constraints
# =========================

# (a) Fleet availability by type
for k in types:
    model.addConstr(
        gp.quicksum(x[i, k] for i in flights) <= aircraft_supply[k],
        name=f"FleetSupply_{k}"
    )

# (b) Max same-model aircraft per flight
for i in flights:
    for k in types:
        model.addConstr(
            x[i, k] <= max_same_model_per_flight,
            name=f"MaxPerFlight_f{i+1}_{k}"
        )

# (c) Demand satisfaction for each flight
for i in flights:
    model.addConstr(
        capacity['A'] * x[i, 'A'] +
        capacity['B'] * x[i, 'B'] +
        capacity['C'] * x[i, 'C'] >= Table_2_Flight_Demand[i],
        name=f"Demand_f{i+1}"
    )

# (d) Flight(s) that must use large aircraft: here modeled as "no C on those flights"
for f in flights_require_large:
    idx = f - 1  # flights are 1-based in data, 0-based in index
    model.addConstr(
        x[idx, 'C'] == 0,
        name=f"NoC_on_f{f}"
    )

# (e) Flights where B is forbidden
for f in forbidden_B_flights:
    idx = f - 1
    model.addConstr(
        x[idx, 'B'] == 0,
        name=f"NoB_on_f{f}"
    )

# =========================
# 6. Solve the model
# =========================

model.optimize()

# =========================
# 7. Print results
# =========================

if model.status == GRB.OPTIMAL:
    print(f"Optimal objective value (total cost in 10,000 yuan): {model.objVal:.4f}")
    print("Optimal aircraft assignment (x[i,k] = number of aircraft type k on flight i):")
    for i in flights:
        for k in types:
            val = x[i, k].X
            if abs(val) > 1e-6:
                print(f"  Flight {i+1}, Type {k}: {val:.0f}")
else:
    print("No optimal solution found.")

# Final answer required: just the optimal total cost value
final_answer_value = model.objVal if model.status == GRB.OPTIMAL else float('nan')
print(f"FinalAnswer=【{final_answer_value}】")