import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and any other necessary packages.
# (Imported above)

# 2. Define all parameter matrices and data inputs.
# Parameters List provided
max_nursing_homes = 4
# Residents list (Index 0 is a placeholder for 1-based indexing alignment)
residents = [0.0, 5.2, 4.4, 7.1, 9.0, 6.1, 5.7, 10.0, 12.2, 7.6, 20.3, 30.4, 30.9, 12.0, 9.3, 15.5, 25.6, 11.0, 5.3, 7.9, 9.9]

num_regions = 20
num_sites = 10

# Coverage mapping based on the problem description and Mermaid diagram
# Key: Region Index (i), Value: List of Site Indices (j) that cover Region i
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
# x[j]: 1 if nursing home at candidate site j is built, 0 otherwise
x = model.addVars(range(1, num_sites + 1), vtype=GRB.BINARY, name="x")

# y[i]: 1 if region i is covered by at least one built nursing home, 0 otherwise
y = model.addVars(range(1, num_regions + 1), vtype=GRB.BINARY, name="y")

# 5. Set up the objective function.
# Minimize the total population not covered: Sum(p_i * (1 - y_i))
# residents[i] corresponds to population of region i
objective_expression = gp.quicksum(residents[i] * (1 - y[i]) for i in range(1, num_regions + 1))
model.setObjective(objective_expression, GRB.MINIMIZE)

# 6. Add all constraints.

# Constraint 1: Budget constraint (At most 4 nursing homes)
model.addConstr(gp.quicksum(x[j] for j in range(1, num_sites + 1)) <= max_nursing_homes, name="BudgetConstraint")

# Constraint 2: Coverage definition
# Region i is covered (y_i = 1) only if at least one site j that serves it is selected (sum(x_j) >= 1).
# y_i <= sum(x_j for j in potential_sites)
# If no serving site is built, sum(x_j) is 0, forcing y_i to 0.
# If >=1 serving site is built, y_i can be 1 (and will be, to minimize the objective penalty).
for i in range(1, num_regions + 1):
    potential_sites = coverage_map[i]
    model.addConstr(y[i] <= gp.quicksum(x[j] for j in potential_sites), name=f"Coverage_Region_{i}")

# 7. Solve the model and print results.
model.optimize()

# Output the answer
if model.status == GRB.OPTIMIZED:
    # Get the objective value (Total Uncovered Population)
    min_uncovered_population = model.ObjVal
    
    print("\nOptimization Results:")
    print(f"Minimum Uncovered Population: {min_uncovered_population} (10,000 people)")
    
    selected_sites = [j for j in range(1, num_sites + 1) if x[j].X > 0.5]
    print(f"Selected Nursing Home Sites: {selected_sites}")
    
    # Final Answer format
    print(f"FinalAnswer=【{min_uncovered_population}】")
else:
    print("No optimal solution found.")