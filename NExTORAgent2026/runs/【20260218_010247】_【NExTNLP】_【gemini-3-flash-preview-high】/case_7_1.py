import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs
empirical_exponent = 1.2
# Table_1_CustomerData format: [Customer ID, Coordinate x, Coordinate y, Monthly shipment]
Table_1_CustomerData = [
    [1, 5, 10, 200],
    [2, 10, 5, 150],
    [3, 0, 12, 200],
    [4, 12, 0, 300]
]

# Create Gurobi model
model = gp.Model("Lemiaoan_Warehouse_Location")

# Set NonConvex parameter for non-linear power functions
model.Params.NonConvex = 2

# 2. Create decision variables
# x-coordinate and y-coordinate of the warehouse location
xw = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="xw")
yw = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="yw")

# 3. Create auxiliary substitution variables
# List to store empirical turnover (c_i) for each customer to set up the objective
ci_list = []

for customer_data in Table_1_CustomerData:
    customer_id, xi, yi, vi = customer_data
    
    # Introduce auxiliary variables with range (-infinity, +infinity)
    dx = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"dx_{customer_id}")
    dy = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"dy_{customer_id}")
    dx2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"dx2_{customer_id}")
    dy2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"dy2_{customer_id}")
    d2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"d2_{customer_id}")
    dp = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"dp_{customer_id}")
    ci = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"ci_{customer_id}")
    
    # 4. Add Constraints
    # Define coordinate differences
    model.addConstr(dx == xw - xi, name=f"dx_constr_{customer_id}")
    model.addConstr(dy == yw - yi, name=f"dy_constr_{customer_id}")
    
    # Define squares: dx2 = dx^2 and dy2 = dy^2
    model.addGenConstrPow(dx, dx2, 2, name=f"dx2_pow_{customer_id}")
    model.addGenConstrPow(dy, dy2, 2, name=f"dy2_pow_{customer_id}")
    
    # d2 = dx2 + dy2 (squared distance)
    model.addConstr(d2 == dx2 + dy2, name=f"d2_sum_{customer_id}")
    
    # Empirical power term: dp = d^1.2. Since d2 = d^2, dp = (d2)^0.6
    # Note: 1.2 / 2 = 0.6
    model.addGenConstrPow(d2, dp, 0.6, name=f"dp_pow_{customer_id}")
    
    # Empirical turnover: ci = vi * dp
    model.addConstr(ci == vi * dp, name=f"ci_def_{customer_id}")
    
    ci_list.append(ci)

# 5. Set up the objective function
# Minimize total empirical turnover Z = sum(ci)
total_empirical_turnover = gp.quicksum(ci_list)
model.setObjective(total_empirical_turnover, GRB.MINIMIZE)

# 6. Solve the model
model.optimize()

# 7. Print results
if model.status == GRB.OPTIMAL:
    min_turnover = model.ObjVal
    print(f"Optimal Warehouse Location: x = {xw.X}, y = {yw.X}")
    print(f"Minimum Total Empirical Turnover: {min_turnover}")
    print(f"FinalAnswer=【{min_turnover}】")
else:
    print("Optimization was not successful.")