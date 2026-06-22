import gurobipy as gp
from gurobipy import GRB

# 1. Initialize Gurobi Model
model = gp.Model("WarehouseLocation")

# Set NonConvex parameter to 2 because we use General Constraints (Norm and Pow)
# which are treated as nonlinear constraints in Gurobi.
model.Params.NonConvex = 2

# 2. Define Parameters
# Using the exact variable names and values from the Parameters List
empirical_exponent = 1.2
Table_1_CustomerData = [
    [1, 5, 10, 200],
    [2, 10, 5, 150],
    [3, 0, 12, 200],
    [4, 12, 0, 300]
]

# 3. Create Decision Variables
# Warehouse location coordinates x_w and y_w
# These are free variables (can be any real number), so we set lb=-GRB.INFINITY
x_w = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x_w")
y_w = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y_w")

# 4. Auxiliary Variables and Constraints construction
# We need to construct the term: shipment * (distance)^1.2 for each customer
objective_expr = 0

for row in Table_1_CustomerData:
    c_id = row[0]
    c_x = row[1]
    c_y = row[2]
    c_vol = row[3]
    
    # Auxiliary variables for coordinate differences: dx = x_w - c_x, dy = y_w - c_y
    # These must be free variables (-inf to +inf)
    dx = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"dx_{c_id}")
    dy = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"dy_{c_id}")
    
    # Link difference variables to warehouse coordinates
    model.addConstr(dx == x_w - c_x, name=f"link_dx_{c_id}")
    model.addConstr(dy == y_w - c_y, name=f"link_dy_{c_id}")
    
    # Distance variable d_i >= 0
    d = model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"d_{c_id}")
    
    # Constraint: d = sqrt(dx^2 + dy^2)  =>  d = norm([dx, dy], 2)
    # This calculates the Euclidean distance using Gurobi's general norm constraint
    model.addGenConstrNorm(d, [dx, dy], 2.0, name=f"norm_dist_{c_id}")
    
    # Empirical turnover factor variable u_i = d_i ^ 1.2
    u = model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"u_{c_id}")
    
    # Constraint: u = d ^ 1.2
    # Syntax: addGenConstrPow(xvar, yvar, a) => yvar = xvar^a
    model.addGenConstrPow(d, u, empirical_exponent, name=f"pow_turnover_{c_id}")
    
    # Add term to objective: volume * u
    objective_expr += c_vol * u

# 5. Set Objective Function
# Minimizing the total empirical turnover
model.setObjective(objective_expr, GRB.MINIMIZE)

# 6. Add Constraints
# (Constraints were added iteratively inside the loop in Step 4)

# 7. Solve the model
model.optimize()

# 8. Output Results
if model.status == GRB.OPTIMAL:
    print(f"Optimal Warehouse Location: x={x_w.X:.4f}, y={y_w.X:.4f}")
    print(f"Minimum Total Empirical Turnover: {model.ObjVal:.4f}")
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimization was not successful.")