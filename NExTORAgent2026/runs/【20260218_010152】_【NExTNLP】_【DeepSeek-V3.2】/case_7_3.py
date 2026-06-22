import gurobipy as gp

# Model creation
model = gp.Model("WarehouseLocation")

# Set non-convex parameter for nonlinear constraints
model.Params.NonConvex = 2

# Extract parameters
empirical_exponent = 1.2
customer_data = [[1, 5, 10, 200], [2, 10, 5, 150], [3, 0, 12, 200], [4, 12, 0, 300]]

# Create dictionaries to store data
x_coords = {}
y_coords = {}
shipments = {}
for i, x, y, v in customer_data:
    x_coords[i] = x
    y_coords[i] = y
    shipments[i] = v

# Create decision variables
x_w = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="x_w")
y_w = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="y_w")

# Create distance variables d_i ≥ 0
d = {}
for i in range(1, 5):
    d[i] = model.addVar(lb=0, ub=gp.GRB.INFINITY, name=f"d_{i}")

# Create empirical turnover variables c_i ≥ 0
c = {}
for i in range(1, 5):
    c[i] = model.addVar(lb=0, ub=gp.GRB.INFINITY, name=f"c_{i}")

# Add constraints for each customer
for i in range(1, 5):
    # Get customer data
    x_i = x_coords[i]
    y_i = y_coords[i]
    v_i = shipments[i]
    
    # Create auxiliary variables for distance calculation
    # xdiff = x_w - x_i, ydiff = y_w - y_i
    xdiff = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name=f"xdiff_{i}")
    ydiff = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name=f"ydiff_{i}")
    
    # xdiff_sq = (x_w - x_i)^2, ydiff_sq = (y_w - y_i)^2
    xdiff_sq = model.addVar(lb=0, ub=gp.GRB.INFINITY, name=f"xdiff_sq_{i}")
    ydiff_sq = model.addVar(lb=0, ub=gp.GRB.INFINITY, name=f"ydiff_sq_{i}")
    
    # sq_sum = (x_w - x_i)^2 + (y_w - y_i)^2
    sq_sum = model.addVar(lb=0, ub=gp.GRB.INFINITY, name=f"sq_sum_{i}")
    
    # d_power = d_i^1.2
    d_power = model.addVar(lb=0, ub=gp.GRB.INFINITY, name=f"d_power_{i}")
    
    # Add constraints
    
    # xdiff = x_w - x_i
    model.addConstr(xdiff == x_w - x_i, name=f"xdiff_constr_{i}")
    
    # ydiff = y_w - y_i
    model.addConstr(ydiff == y_w - y_i, name=f"ydiff_constr_{i}")
    
    # xdiff_sq = xdiff^2
    model.addGenConstrPow(xdiff, xdiff_sq, 2, name=f"xdiff_pow_{i}")
    
    # ydiff_sq = ydiff^2
    model.addGenConstrPow(ydiff, ydiff_sq, 2, name=f"ydiff_pow_{i}")
    
    # sq_sum = xdiff_sq + ydiff_sq
    model.addConstr(sq_sum == xdiff_sq + ydiff_sq, name=f"sq_sum_constr_{i}")
    
    # d_i = sqrt(sq_sum)  [d_i^2 = sq_sum]
    model.addGenConstrPow(d[i], sq_sum, 0.5, name=f"dist_constr_{i}")
    
    # d_power = d_i^1.2
    model.addGenConstrPow(d[i], d_power, empirical_exponent, name=f"power_constr_{i}")
    
    # c_i = v_i * d_i^1.2
    model.addConstr(c[i] == v_i * d_power, name=f"turnover_constr_{i}")

# Set objective: minimize total empirical turnover
model.setObjective(c[1] + c[2] + c[3] + c[4], gp.GRB.MINIMIZE)

# Solve the model
model.optimize()

# Print results and final answer
if model.status == gp.GRB.OPTIMAL:
    print(f"Optimal warehouse location: ({x_w.x:.6f}, {y_w.y:.6f})")
    print(f"Optimal total empirical turnover: {model.objVal:.6f}")
    
    # Output the answer
    print(f"FinalAnswer=【{model.objVal:.6f}】")
else:
    print(f"Model not solved to optimality. Status: {model.status}")
    print(f"FinalAnswer=【None】")