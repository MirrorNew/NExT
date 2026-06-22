import gurobipy as gp
import math

# Parameters from the provided list
Q = [10, 15, 20, 25]
coordinates = [[1, 1], [1, 2], [2, 1], [2, 2]]
number_of_work_sites = 4

# Create model
model = gp.Model("ConcreteMixingStationLocation")

# Allow non-convex constraints
model.Params.NonConvex = 2

# Decision variables for the mixing station location
x = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="x")
y = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="y")

# Auxiliary variables for squared distances and distances
d_sq = []
d = []

for i in range(number_of_work_sites):
    # Squared distance variable
    d_sq_i = model.addVar(lb=0, ub=gp.GRB.INFINITY, name=f"d_sq_{i+1}")
    d_sq.append(d_sq_i)
    
    # Distance variable
    d_i = model.addVar(lb=0, ub=gp.GRB.INFINITY, name=f"d_{i+1}")
    d.append(d_i)
    
    # Constraint: d_sq_i = (x - x_i)^2 + (y - y_i)^2
    x_i, y_i = coordinates[i]
    model.addConstr(d_sq_i == (x - x_i)**2 + (y - y_i)**2, name=f"sq_dist_{i+1}")
    
    # General constraint: d_i = sqrt(d_sq_i)
    model.addGenConstrPow(d_i, d_sq_i, 0.5, name=f"sqrt_{i+1}")

# Objective: minimize total transportation volume
obj = gp.quicksum(Q[i] * d[i] for i in range(number_of_work_sites))
model.setObjective(obj, gp.GRB.MINIMIZE)

# Solve the model
model.optimize()

# Print results
print("\n--- Optimization Results ---")
print(f"Status: {model.status}")
if model.status == gp.GRB.OPTIMAL:
    print(f"Optimal objective value: {model.objVal:.4f}")
    print(f"Location of mixing station: (x, y) = ({x.X:.4f}, {y.X:.4f})")
    
    print("\nDetailed distances:")
    total_transport = 0
    for i in range(number_of_work_sites):
        dist = d[i].X
        transport = Q[i] * dist
        total_transport += transport
        x_i, y_i = coordinates[i]
        print(f"  Work site {i+1} at ({x_i}, {y_i}): distance = {dist:.4f}, "
              f"transport volume = {transport:.4f}")
    
    print(f"\nTotal transportation volume: {total_transport:.4f}")
    
    # Output final answer as requested
    print(f"\nFinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found.")
    print("FinalAnswer=【None】")