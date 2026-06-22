import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and any other necessary packages.
# (Imported above)

# 2. Define all parameter matrices and data inputs.
# Parameter: Max number of nursing homes to build
max_nursing_homes = 4

# Parameter: Residents in each region (10,000 people)
# Index 0 is a dummy value so that index i corresponds to Region i (1-20)
residents = [0.0, 5.2, 4.4, 7.1, 9.0, 6.1, 5.7, 10.0, 12.2, 7.6, 20.3, 30.4, 30.9, 12.0, 9.3, 15.5, 25.6, 11.0, 5.3, 7.9, 9.9]

num_regions = 20
num_sites = 10

# Parameter: Coverage mapping
# This dictionary maps each Region i (Key) to the list of Nursing Home Sites j (Value) that can serve it.
# Derived from the "Regional distribution map" in the problem.
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

# Create the model
model = gp.Model("ElderlyCareFacilityOptimization")

# 3. Create decision variables.
# x[j]: Binary variable, 1 if nursing home at site j is built, 0 otherwise.
# Range: j = 1 to 10
x = model.addVars(range(1, num_sites + 1), vtype=GRB.BINARY, name="x")

# y[i]: Binary variable, 1 if region i is covered, 0 otherwise.
# Range: i = 1 to 20
y = model.addVars(range(1, num_regions + 1), vtype=GRB.BINARY, name="y")

# 5. Set up the objective function.
# Minimize the total population NOT covered.
# Objective = Sum(p_i * (1 - y_i)) for all regions i
objective_expression = gp.quicksum(residents[i] * (1 - y[i]) for i in range(1, num_regions + 1))
model.setObjective(objective_expression, GRB.MINIMIZE)

# 6. Add all constraints.

# Constraint 1: Budget Constraint
# The total number of built nursing homes cannot exceed max_nursing_homes (4).
model.addConstr(gp.quicksum(x[j] for j in range(1, num_sites + 1)) <= max_nursing_homes, name="BudgetConstraint")

# Constraint 2: Coverage Definition
# A region i can be covered (y[i]=1) only if at least one serving site j is built.
# y_i <= sum(x_j for j in Potential_Sites_for_i)
# If sum(x_j) is 0, y_i must be 0.
# The minimization objective will force y_i to 1 whenever possible (since 1-1=0 cost) if the constraint allows it.
for i in range(1, num_regions + 1):
    potential_sites = coverage_map[i]
    model.addConstr(y[i] <= gp.quicksum(x[j] for j in potential_sites), name=f"Coverage_Region_{i}")

# 7. Solve the model and print results.
model.optimize()

if model.status == GRB.OPTIMIZED:
    # Get the minimum uncovered population
    min_uncovered_population = model.ObjVal
    
    # Identify selected sites
    selected_sites = [j for j in range(1, num_sites + 1) if x[j].X > 0.5]
    
    print("\nOptimization Successful:")
    print(f"Minimum Uncovered Population: {min_uncovered_population} (10,000 people)")
    print(f"Selected Nursing Home Locations: {selected_sites}")
    
    # Final Answer output as required
    print(f"FinalAnswer=【{min_uncovered_population}】")
else:
    print("No optimal solution found.")