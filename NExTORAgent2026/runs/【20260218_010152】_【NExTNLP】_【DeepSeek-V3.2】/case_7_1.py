import gurobipy as gp

# Model creation
model = gp.Model("WarehouseLocation")

# Set non-convex parameter for nonlinear constraints
model.Params.NonConvex = 2

# Extract parameters from Parameters List
empirical_exponent = 1.2
customer_data = [[1, 5, 10, 200], [2, 10, 5, 150], [3, 0, 12, 200], [4, 12, 0, 300]]

# Create dictionaries to store customer data
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
    
    # Create auxiliary variables for implementing mathematical constraints
    
    # xdiff_i = x_w - x_i
    xdiff_i = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name=f"xdiff_{i}")
    
    # ydiff_i = y_w - y_i
    ydiff_i = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name=f"ydiff_{i}")
    
    # xdiff_sq_i = (x_w - x_i)^2
    xdiff_sq_i = model.addVar(lb=0, ub=gp.GRB.INFINITY, name=f"xdiff_sq_{i}")
    
    # ydiff_sq_i = (y_w - y_i)^2
    ydiff_sq_i = model.addVar(lb=0, ub=gp.GRB.INFINITY, name=f"ydiff_sq_{i}")
    
    # sq_i = (x_w - x_i)^2 + (y_w - y_i)^2
    sq_i = model.addVar(lb=0, ub=gp.GRB.INFINITY, name=f"sq_{i}")
    
    # p_i = d_i^1.2
    p_i = model.addVar(lb=0, ub=gp.GRB.INFINITY, name=f"p_{i}")
    
    # Add constraints
    
    # xdiff_i = x_w - x_i
    model.addConstr(xdiff_i == x_w - x_i, name=f"xdiff_constr_{i}")
    
    # ydiff_i = y_w - y_i
    model.addConstr(ydiff_i == y_w - y_i, name=f"ydiff_constr_{i}")
    
    # xdiff_sq_i = xdiff_i^2
    model.addGenConstrPow(xdiff_i, xdiff_sq_i, 2, name=f"xdiff_pow_{i}")
    
    # ydiff_sq_i = ydiff_i^2
    model.addGenConstrPow(ydiff_i, ydiff_sq_i, 2, name=f"ydiff_pow_{i}")
    
    # sq_i = xdiff_sq_i + ydiff_sq_i
    model.addConstr(sq_i == xdiff_sq_i + ydiff_sq_i, name=f"sq_constr_{i}")
    
    # d_i = sqrt(sq_i)
    model.addGenConstrPow(d[i], sq_i, 0.5, name=f"dist_constr_{i}")
    
    # p_i = d_i^1.2
    model.addGenConstrPow(d[i], p_i, empirical_exponent, name=f"power_constr_{i}")
    
    # c_i = v_i * p_i
    model.addConstr(c[i] == v_i * p_i, name=f"turnover_constr_{i}")

# Set objective: minimize total empirical turnover
model.setObjective(c[1] + c[2] + c[3] + c[4], gp.GRB.MINIMIZE)

# Solve the model
model.optimize()

# Print results and final answer
if model.status == gp.GRB.OPTIMAL:
    print(f"Optimal warehouse location: ({x_w.x:.6f}, {y_w.y:.6f})")
    print(f"Optimal total empirical turnover: {model.objVal:.6f}")
    
    # Output the answer as required
    print(f"FinalAnswer=【{model.objVal:.6f}】")
else:
    print(f"Model not solved to optimality. Status: {model.status}")
    print(f"FinalAnswer=【None】")