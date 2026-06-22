import gurobipy as gp
from gurobipy import GRB

# 1. Initialize Gurobi Model
model = gp.Model("WarehouseLocation")

# Set NonConvex parameter to 2 because we use General Constraints (Norm and Pow) 
# which are treated as nonlinear constraints.
model.Params.NonConvex = 2

# 2. Define Parameters
# Empirical exponent alpha = 1.2
empirical_exponent = 1.2

# Customer Data: [ID, Coordinate x, Coordinate y, Monthly shipment]
# Matches Table 6-2 in the problem description
customers_data = [
    [1, 5, 10, 200],
    [2, 10, 5, 150],
    [3, 0, 12, 200],
    [4, 12, 0, 300]
]

# 3. Create Decision Variables
# Warehouse location (x_w, y_w).
# These are free variables (range -infinity to +infinity).
x_w = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x_w")
y_w = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y_w")

# 4. Create Auxiliary Variables and Constraints
# We build the objective expression iteratively.
objective_expr = 0

for cust in customers_data:
    c_id, c_x, c_y, c_vol = cust
    
    # Auxiliary variables for coordinate differences: dx = x_w - c_x, dy = y_w - c_y
    # These must allow negative values, so lb=-GRB.INFINITY.
    dx = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"dx_{c_id}")
    dy = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"dy_{c_id}")
    
    # Constraints defining the differences
    model.addConstr(dx == x_w - c_x, name=f"def_dx_{c_id}")
    model.addConstr(dy == y_w - c_y, name=f"def_dy_{c_id}")
    
    # Auxiliary variable for Euclidean distance: d_i
    # Distance is non-negative.
    d = model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"d_{c_id}")
    
    # Constraint: d_i = sqrt(dx_i^2 + dy_i^2) => d_i = Norm([dx_i, dy_i], 2)
    model.addGenConstrNorm(d, [dx, dy], 2.0, name=f"norm_dist_{c_id}")
    
    # Auxiliary variable for the empirical turnover term: u_i = d_i ^ 1.2
    # This is the "empirical correction" applied to the distance.
    u = model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"u_{c_id}")
    
    # Constraint: u_i = d_i ^ 1.2 using addGenConstrPow(x, y, a) => y = x^a
    model.addGenConstrPow(d, u, empirical_exponent, name=f"pow_term_{c_id}")
    
    # Accumulate objective: Volume * EmpiricalDistance
    objective_expr += c_vol * u

# 5. Set Objective Function
# Minimize the total empirical turnover
model.setObjective(objective_expr, GRB.MINIMIZE)

# 6. Solve the model
model.optimize()

# 7. Print results
if model.status == GRB.OPTIMIZED:
    print(f"Optimal Warehouse Location: x = {x_w.X:.4f}, y = {y_w.X:.4f}")
    print(f"Minimum Total Empirical Turnover: {model.ObjVal:.4f}")
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimization was not successful.")