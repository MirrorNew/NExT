import gurobipy as gp
import math

# Model creation
model = gp.Model("WarehouseLocation")

# Set non-convex parameter
model.Params.NonConvex = 2

# Extract parameters
empirical_exponent = 1.2
customer_data = [[1, 5, 10, 200], [2, 10, 5, 150], [3, 0, 12, 200], [4, 12, 0, 300]]

# Create decision variables
x_w = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="x_w")
y_w = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="y_w")

# Create distance variables
d = {}
for i in range(1, 5):
    d[i] = model.addVar(lb=0, ub=gp.GRB.INFINITY, name=f"d_{i}")

# Create empirical turnover variables
c = {}
for i in range(1, 5):
    c[i] = model.addVar(lb=0, ub=gp.GRB.INFINITY, name=f"c_{i}")

# Add constraints for distances and empirical turnovers
for i, x_i, y_i, v_i in customer_data:
    # Distance constraint: d_i = sqrt((x_w - x_i)^2 + (y_w - y_i)^2)
    # This requires creating auxiliary variables for the squared terms
    
    # Create auxiliary variables for squared differences
    xdiff_sq = model.addVar(lb=0, ub=gp.GRB.INFINITY, name=f"xdiff_sq_{i}")
    ydiff_sq = model.addVar(lb=0, ub=gp.GRB.INFINITY, name=f"ydiff_sq_{i}")
    sq_sum = model.addVar(lb=0, ub=gp.GRB.INFINITY, name=f"sq_sum_{i}")
    
    # Add constraints for squared differences
    # (x_w - x_i)^2 = xdiff_sq
    xdiff = x_w - x_i
    model.addGenConstrPow(xdiff, xdiff_sq, 2, name=f"xdiff_pow_{i}")
    
    # (y_w - y_i)^2 = ydiff_sq
    ydiff = y_w - y_i
    model.addGenConstrPow(ydiff, ydiff_sq, 2, name=f"ydiff_pow_{i}")
    
    # Sum of squares
    model.addConstr(sq_sum == xdiff_sq + ydiff_sq, name=f"sq_sum_constr_{i}")
    
    # Distance = sqrt(sq_sum)
    model.addGenConstrPow(d[i], sq_sum, 0.5, name=f"dist_pow_{i}")
    
    # Create auxiliary variable for d_i^1.2
    d_power = model.addVar(lb=0, ub=gp.GRB.INFINITY, name=f"d_power_{i}")
    model.addGenConstrPow(d[i], d_power, empirical_exponent, name=f"power_constr_{i}")
    
    # Empirical turnover: c_i = v_i * d_i^1.2
    model.addConstr(c[i] == v_i * d_power, name=f"c_constr_{i}")

# Set objective: minimize total empirical turnover
model.setObjective(c[1] + c[2] + c[3] + c[4], gp.GRB.MINIMIZE)

# Solve the model
model.optimize()

# Print results
if model.status == gp.GRB.OPTIMAL:
    print(f"Optimal warehouse location: ({x_w.x:.4f}, {y_w.y:.4f})")
    print(f"Optimal total empirical turnover: {model.objVal:.4f}")
    
    # Calculate the answer as the objective value (total empirical turnover)
    final_answer = model.objVal
    print(f"FinalAnswer=【{final_answer:.4f}】")
else:
    print(f"Model not solved to optimality. Status: {model.status}")
    final_answer = None
    print(f"FinalAnswer=【{final_answer}】")