import gurobipy as gp
from gurobipy import GRB

# ============================
# 1. Define parameters (use ONLY the given Parameters List)
# ============================

empirical_exponent = 1.2  # From Parameters List

# Table_1_CustomerData: [CustomerID, x_i, y_i, v_i]
Table_1_CustomerData = [
    [1, 5, 10, 200],
    [2, 10, 5, 150],
    [3, 0, 12, 200],
    [4, 12, 0, 300]
]

# Extract structured data
customer_ids = [row[0] for row in Table_1_CustomerData]
x_i_data = {row[0]: row[1] for row in Table_1_CustomerData}
y_i_data = {row[0]: row[2] for row in Table_1_CustomerData}
v_i_data = {row[0]: row[3] for row in Table_1_CustomerData}

# ============================
# 2. Create model
# ============================

model = gp.Model("Nonlinear_Warehouse_Location")
model.Params.NonConvex = 2  # Required for general nonlinear constraints (Pow)

# ============================
# 3. Decision variables
# ============================

# Warehouse location
x_w = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x_w")
y_w = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y_w")

# Distances d_i >= 0
d = model.addVars(customer_ids, lb=0.0, vtype=GRB.CONTINUOUS, name="d")

# Empirical turnover c_i >= 0
c = model.addVars(customer_ids, lb=0.0, vtype=GRB.CONTINUOUS, name="c")

# ============================
# 4. Auxiliary substitution variables
#    (all with lb=-INF, ub=+INF as required)
# ============================

# dx_i, dy_i: coordinate differences
dx = model.addVars(customer_ids, lb=-GRB.INFINITY, ub=GRB.INFINITY,
                   vtype=GRB.CONTINUOUS, name="dx")
dy = model.addVars(customer_ids, lb=-GRB.INFINITY, ub=GRB.INFINITY,
                   vtype=GRB.CONTINUOUS, name="dy")

# s_i: squared distance = dx_i^2 + dy_i^2
s = model.addVars(customer_ids, lb=-GRB.INFINITY, ub=GRB.INFINITY,
                  vtype=GRB.CONTINUOUS, name="s")

# p_i: d_i^1.2
p = model.addVars(customer_ids, lb=-GRB.INFINITY, ub=GRB.INFINITY,
                  vtype=GRB.CONTINUOUS, name="p")

# ============================
# 5. Objective function
#    Minimize total empirical turnover = sum_i c_i
# ============================

model.setObjective(gp.quicksum(c[i] for i in customer_ids), GRB.MINIMIZE)

# ============================
# 6. Constraints
# ============================

for i in customer_ids:
    # 6.1 Linear relations for dx_i, dy_i
    model.addConstr(dx[i] == x_w - x_i_data[i], name=f"dx_def_{i}")
    model.addConstr(dy[i] == y_w - y_i_data[i], name=f"dy_def_{i}")

    # 6.2 Quadratic relation for squared distance s_i = dx_i^2 + dy_i^2
    model.addConstr(s[i] == dx[i] * dx[i] + dy[i] * dy[i], name=f"s_def_{i}")

    # 6.3 Link distance and squared distance: d_i^2 = s_i
    #     Using general power constraint: Y = X^2  =>  X: d[i], Y: s[i]
    model.addGenConstrPow(d[i], s[i], 2.0, name=f"d_sq_{i}")

    # 6.4 p_i = d_i^1.2 (empirical_exponent)
    #     Using general power constraint: Y = X^1.2  =>  X: d[i], Y: p[i]
    model.addGenConstrPow(d[i], p[i], empirical_exponent, name=f"d_pow_{i}")

    # 6.5 Empirical turnover c_i = v_i * p_i (linear)
    model.addConstr(c[i] == v_i_data[i] * p[i], name=f"c_def_{i}")

# ============================
# 7. Solve the model
# ============================

model.optimize()

# ============================
# 8. Print results
# ============================

if model.status == GRB.OPTIMAL:
    x_w_val = x_w.X
    y_w_val = y_w.X
    Z_opt = model.ObjVal

    print("Optimal solution found:")
    print(f"  Warehouse location: x_w = {x_w_val:.6f}, y_w = {y_w_val:.6f}")
    print(f"  Minimum total empirical turnover Z = {Z_opt:.6f}")
    for i in customer_ids:
        print(f"Customer {i}: d_{i} = {d[i].X:.6f}, c_{i} = {c[i].X:.6f}")
else:
    print(f"Optimization ended with status {model.status}")
    x_w_val = float('nan')
    y_w_val = float('nan')
    Z_opt = float('nan')

# ============================
# 9. FinalAnswer output
#     The question asks: determine the warehouse location that minimizes
#     total empirical turnover. We output the coordinates as the answer.
# ============================

print(f"FinalAnswer=【({x_w_val:.6f}, {y_w_val:.6f})】")