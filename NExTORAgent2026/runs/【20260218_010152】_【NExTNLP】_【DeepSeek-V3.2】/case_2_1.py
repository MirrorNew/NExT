import gurobipy as gp

# 1. Create model
model = gp.Model("BeamCrossSectionOptimization")

# 2. Define parameters
r = 300
sigma_y = 100  # Assumed yield stress value (needed for constraint, can be adjusted)

# 3. Create decision variables
x1 = model.addVar(lb=0, ub=r, name="x1")        # half-width
x2 = model.addVar(lb=0, ub=r, name="x2")        # half-height
M = model.addVar(lb=0, name="M")                # bending moment to maximize
sigma_max = model.addVar(lb=0, name="sigma_max") # maximum bending stress

# 4. Create auxiliary substitution variables
y2 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="y2")  # y2 = x2^2
y1 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="y1")  # y1 = x1 * y2 = x1 * x2^2

# 5. Set objective function
model.setObjective(M, gp.GRB.MAXIMIZE)

# 6. Add constraints
# Enable non-convex mode for power constraint
model.Params.NonConvex = 2

# Circular billet boundary: x1^2 + x2^2 <= r^2
model.addConstr(x1 * x1 + x2 * x2 <= r * r, name="circular_boundary")

# Auxiliary constraints for denominator elimination
model.addGenConstrPow(x2, y2, 2, name="x2_squared")  # y2 = x2^2
model.addConstr(x1 * y2 == y1, name="x1_times_x2_squared")  # y1 = x1 * y2

# Maximum stress definition: sigma_max = (3/4)*M/(x1*x2^2) = (3/4)*M/y1
# Rearrange to avoid division: (3/4)*M = sigma_max * y1
model.addConstr(0.75 * M == sigma_max * y1, name="stress_definition")

# Yield stress limit: sigma_max <= sigma_y
model.addConstr(sigma_max <= sigma_y, name="yield_limit")

# 7. Solve the model
model.optimize()

# Print results
if model.status == gp.GRB.OPTIMAL:
    print("Optimal solution found:")
    print(f"x1 (half-width) = {x1.X:.4f}")
    print(f"x2 (half-height) = {x2.X:.4f}")
    print(f"M (maximum bending moment) = {M.X:.4f}")
    print(f"sigma_max = {sigma_max.X:.4f}")
    print(f"Check: x1^2 + x2^2 = {x1.X**2 + x2.X**2:.4f} <= r^2 = {r**2}")
    print(f"Check: sigma_max = {sigma_max.X:.4f} <= sigma_y = {sigma_y}")
    
    # Output final answer as requested
    print(f"FinalAnswer=【{x1.X}】")
else:
    print(f"Optimization failed. Status: {model.status}")
    print(f"FinalAnswer=【None】")