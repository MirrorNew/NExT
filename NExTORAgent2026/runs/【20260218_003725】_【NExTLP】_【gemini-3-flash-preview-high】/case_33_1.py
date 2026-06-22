import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
max_nursing_homes = 4
residents = [0.0, 5.2, 4.4, 7.1, 9.0, 6.1, 5.7, 10.0, 12.2, 7.6, 20.3, 30.4, 30.9, 12.0, 9.3, 15.5, 25.6, 11.0, 5.3, 7.9, 9.9]

# Coverage mapping: Region index (1-20) -> list of candidate nursing home sites (1-10)
# This mapping is derived from the "A -> P" arrows in Figure C-13 and problem examples
coverage_map = {
    1: [2],
    2: [1, 2],
    3: [1, 3],
    4: [3],
    5: [3],
    6: [2],
    7: [2, 4],
    8: [3, 4],
    9: [8],
    10: [4, 6],
    11: [4, 5],
    12: [4, 5, 6],
    13: [4, 5, 7],
    14: [8, 9],
    15: [6, 9],
    16: [5, 6],
    17: [5, 7, 10],
    18: [8, 9],
    19: [9, 10],
    20: [10]
}

# 2. Create the model
model = gp.Model("Sunset_Red_Optimization")

# 3. Create decision variables
# x[j] = 1 if nursing home j is built, else 0 (j=1..10)
x = model.addVars(range(1, 11), vtype=GRB.BINARY, name="x")
# y[i] = 1 if region i is covered by at least one built nursing home, else 0 (i=1..20)
y = model.addVars(range(1, 21), vtype=GRB.BINARY, name="y")

# 4. Set up the objective function
# Minimize the total population not covered: Sum of residents[i] * (1 - y[i])
model.setObjective(gp.quicksum(residents[i] * (1 - y[i]) for i in range(1, 21)), GRB.MINIMIZE)

# 5. Add all constraints
# Budget constraint: At most 4 nursing homes can be set up
model.addConstr(gp.quicksum(x[j] for j in range(1, 11)) <= max_nursing_homes, name="Budget_Constraint")

# Coverage constraints using indicator variables:
# y[i] = 1 if and only if at least one nursing home serving region i is built
for i in range(1, 21):
    site_sum = gp.quicksum(x[j] for j in coverage_map[i])
    # Case: y[i] is 1 (covered)
    model.addGenConstrIndicator(y[i], 1, site_sum >= 1, name=f"Coverage_True_{i}")
    # Case: y[i] is 0 (not covered)
    model.addGenConstrIndicator(y[i], 0, site_sum <= 0, name=f"Coverage_False_{i}")

# 6. Solve the model
model.optimize()

# 7. Solve and print results
if model.status == GRB.OPTIMAL:
    selected_sites = [j for j in range(1, 11) if x[j].X > 0.5]
    print(f"Selected Nursing Home Sites: {selected_sites}")
    print(f"Minimized Total Population Not Covered: {model.ObjVal} (10,000 people)")
    
    # Final output as requested
    print(f"FinalAnswer=【{model.ObjVal}】")