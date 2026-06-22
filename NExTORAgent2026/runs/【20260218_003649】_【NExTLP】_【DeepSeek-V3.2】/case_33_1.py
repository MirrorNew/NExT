import gurobipy as gp

# Parameters
max_nursing_homes = 4
residents = [0.0, 5.2, 4.4, 7.1, 9.0, 6.1, 5.7, 10.0, 12.2, 7.6, 20.3, 30.4, 30.9, 12.0, 9.3, 15.5, 25.6, 11.0, 5.3, 7.9, 9.9]

# Create model
model = gp.Model("NursingHomeLocation")

# Decision variables
x = {}
for j in range(1, 11):
    x[j] = model.addVar(vtype=gp.GRB.BINARY, name=f"x_{j}")

y = {}
for i in range(1, 21):
    y[i] = model.addVar(vtype=gp.GRB.BINARY, name=f"y_{i}")

# Set objective
obj = gp.quicksum(residents[i] * (1 - y[i]) for i in range(1, 21))
model.setObjective(obj, gp.GRB.MINIMIZE)

# Add constraints
# Constraint 1: At most 4 nursing homes can be built
model.addConstr(gp.quicksum(x[j] for j in range(1, 11)) <= max_nursing_homes, "max_nursing_homes")

# Constraint 2: Coverage definition - y_i=1 implies region i is covered by at least one nursing home
# From the mermaid diagram, define service relationships a_ij
service_relations = {
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

# Add indicator constraints for coverage
for i in range(1, 21):
    # Create linear expression for sum of a_ij * x_j
    coverage_expr = gp.quicksum(x[j] for j in service_relations[i])
    # y_i = 1 => coverage_expr >= 1
    model.addGenConstrIndicator(y[i], 1, coverage_expr >= 1, name=f"cover_indicator_{i}")
    # Note: We don't need the reverse (y_i=0 => coverage_expr=0) because the objective will force y_i=0 
    # when not covered, as that minimizes the uncovered population

# Optimize
model.optimize()

# Print solution
if model.status == gp.GRB.OPTIMAL:
    print("Optimal solution found:")
    print(f"Objective value (uncovered population): {model.ObjVal:.2f} (10,000 people)")
    
    # Built nursing homes
    built = [j for j in range(1, 11) if x[j].X > 0.5]
    print(f"\nBuilt nursing homes at sites: {built}")
    
    # Uncovered regions
    uncovered_regions = [i for i in range(1, 21) if y[i].X < 0.5]
    uncovered_population = sum(residents[i] for i in uncovered_regions)
    print(f"\nUncovered regions: {uncovered_regions}")
    print(f"Total uncovered population: {uncovered_population:.2f} (10,000 people)")
    
    # Answer to the question: total uncovered population
    print(f"FinalAnswer=【{model.ObjVal:.2f}】")
else:
    print("No optimal solution found")
    print(f"FinalAnswer=【No feasible solution】")