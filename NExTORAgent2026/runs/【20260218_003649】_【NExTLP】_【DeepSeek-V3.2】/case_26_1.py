import gurobipy as gp

# Define parameters from the given list
part_types = ['A', 'B', 'C']
worker_levels = [1, 2, 3, 4, 5, 6]
hours_per_week = 40
weekly_demand = {'A': 1940, 'B': 1000, 'C': 10060}

# Extract data from Table C-7
# Format: [level, num_workers, wage, hours_A, hours_B, hours_C]
table_c7 = [[1, 4, 15.0, 160, 0, 0],
            [2, 9, 14.5, 360, 0, 0],
            [3, 20, 13.0, 600, 200, 0],
            [4, 54, 12.0, 0, 160, 2000],
            [5, 102, 10.5, 0, 80, 4000],
            [6, 40, 9.75, 0, 0, 1600]]

# Extract data from Training cost table
# Format: [level, cost_A, cost_B, cost_C]
training_cost = [[1, 0, 10, 5],
                 [2, 0, 20, 5],
                 [3, 0, 0, 10],
                 [4, 15, 0, 0],
                 [5, 20, 0, 0],
                 [6, 25, 20, 0]]

# Extract data from Table C-8 (work efficiency)
# Format: [level, efficiency_A, efficiency_B, efficiency_C]
table_c8 = [[1, 2.0, 1.2, 2.0],
            [2, 1.8, 1.08, 1.8],
            [3, 1.62, 2.5, 1.62],
            [4, 1.8, 2.16, 1.45],
            [5, 1.62, 1.93, 1.31],
            [6, 1.3, 1.74, 1.2]]

# Create dictionaries for easy access
N = {row[0]: row[1] for row in table_c7}  # Number of workers per level
w = {row[0]: row[2] for row in table_c7}  # Hourly wage per level

# Training cost dictionary: c[level][part_type]
c = {row[0]: {'A': row[1], 'B': row[2], 'C': row[3]} for row in training_cost}

# Work efficiency dictionary: r[level][part_type]
r = {row[0]: {'A': row[1], 'B': row[2], 'C': row[3]} for row in table_c8}

# Create model
model = gp.Model("Worker_Scheduling_Optimization")

# Decision variables
h = {}  # Weekly working hours: h[level][part_type]
k = {}  # Number of trained workers: k[level][part_type]

for i in worker_levels:
    for j in part_types:
        # h_ij: continuous variable for weekly hours
        h[i, j] = model.addVar(lb=0, ub=40*N[i], name=f"h_{i}_{j}")
        # k_ij: integer variable for number of trained workers
        k[i, j] = model.addVar(lb=0, ub=N[i], vtype=gp.GRB.INTEGER, name=f"k_{i}_{j}")

# Set objective function: minimize total cost (wages + training costs)
obj_expr = gp.QuadExpr()
for i in worker_levels:
    for j in part_types:
        obj_expr += w[i] * h[i, j] + c[i][j] * k[i, j]
model.setObjective(obj_expr, gp.GRB.MINIMIZE)

# Constraints
# 1. Time capacity per worker level
for i in worker_levels:
    model.addConstr(gp.quicksum(h[i, j] for j in part_types) <= 40 * N[i], 
                   name=f"Time_capacity_level_{i}")

# 2. Demand satisfaction
for j in part_types:
    model.addConstr(gp.quicksum(r[i][j] * h[i, j] for i in worker_levels) >= weekly_demand[j],
                   name=f"Demand_{j}")

# 3. Training capacity linking hours to trained workers
for i in worker_levels:
    for j in part_types:
        model.addConstr(h[i, j] <= 40 * k[i, j], 
                       name=f"Training_link_{i}_{j}")

# 4. Training upper bound (already enforced by variable bounds)

# Optimize the model
model.optimize()

# Check if solution is found
if model.status == gp.GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"Total cost (salary expenditure): {model.ObjVal:.2f} yuan")
    
    # Print detailed solution
    print("\nDetailed schedule (hours per week):")
    print("Level | Part A | Part B | Part C | Trained A | Trained B | Trained C")
    print("-" * 70)
    
    total_hours = {j: 0 for j in part_types}
    for i in worker_levels:
        hours_A = h[i, 'A'].X if (i, 'A') in h else 0
        hours_B = h[i, 'B'].X if (i, 'B') in h else 0
        hours_C = h[i, 'C'].X if (i, 'C') in h else 0
        trained_A = k[i, 'A'].X if (i, 'A') in k else 0
        trained_B = k[i, 'B'].X if (i, 'B') in k else 0
        trained_C = k[i, 'C'].X if (i, 'C') in k else 0
        
        print(f"{i:5} | {hours_A:7.1f} | {hours_B:7.1f} | {hours_C:7.1f} | "
              f"{trained_A:9} | {trained_B:9} | {trained_C:9}")
        
        total_hours['A'] += hours_A
        total_hours['B'] += hours_B
        total_hours['C'] += hours_C
    
    print("\nTotal hours allocated:")
    for j in part_types:
        print(f"Part {j}: {total_hours[j]:.1f} hours")
    
    print("\nProduction output (pieces):")
    for j in part_types:
        output = sum(r[i][j] * h[i, j].X for i in worker_levels)
        print(f"Part {j}: {output:.1f} pieces (demand: {weekly_demand[j]})")
    
    # Calculate total training cost
    total_training_cost = sum(c[i][j] * k[i, j].X for i in worker_levels for j in part_types)
    print(f"\nTotal training cost: {total_training_cost:.2f} yuan")
    
    # Calculate total wage cost
    total_wage_cost = sum(w[i] * h[i, j].X for i in worker_levels for j in part_types)
    print(f"Total wage cost: {total_wage_cost:.2f} yuan")
    
    print(f"FinalAnswer=【{model.ObjVal:.2f}】")
else:
    print("No optimal solution found.")
    print(f"Model status: {model.status}")
    print(f"FinalAnswer=【No optimal solution found】")