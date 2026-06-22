import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Define parameters
# =========================

# Parameters List (must use these values)
m = 2  # number of warehouses
n = 2  # number of customers
d = [80, 70]  # customer demands
s_max = [100, 100]  # warehouse max supplies
cost_quad = [[0.01, 0.01],
             [0.02, 0.02]]  # quadratic cost coefficients
cost_lin = [[2.0, 3.0],
            [2.5, 1.5]]     # linear cost coefficients

# =========================
# 2. Create model
# =========================
model = gp.Model("Quadratic_Transportation")

# Allow quadratic / nonconvex constructs (general safety as per instructions)
model.Params.NonConvex = 2

# =========================
# 3. Create decision variables x[i,j] >= 0
#    i: warehouse index (0..m-1), j: customer index (0..n-1)
# =========================
x = model.addVars(m, n, lb=0.0, vtype=GRB.CONTINUOUS, name="x")

# Optional: upper bound 100 consistent with context (not strictly required,
# since supply and demand already bound the flows)
for i in range(m):
    for j in range(n):
        x[i, j].UB = 100.0

# =========================
# 4. (Optional) auxiliary variables for quadratic terms (not strictly needed,
#    but we include them as generic nonlinear modeling practice)
#    y[i,j] = x[i,j]^2
# =========================
y = model.addVars(m, n, lb=-GRB.INFINITY, ub=GRB.INFINITY,
                  vtype=GRB.CONTINUOUS, name="y")

# Define y[i,j] = x[i,j]^2 using general power constraints
for i in range(m):
    for j in range(n):
        model.addGenConstrPow(x[i, j], y[i, j], 2.0, name=f"pow_x_{i}_{j}")

# =========================
# 5. Objective: minimize total transportation cost
#    Z = sum_{i,j} (cost_quad[i][j] * x[i,j]^2 + cost_lin[i][j] * x[i,j])
#    Implemented as linear in y plus linear in x:
#    Z = sum_{i,j} (cost_quad[i][j] * y[i,j] + cost_lin[i][j] * x[i][j])
# =========================
obj_expr = gp.LinExpr()
for i in range(m):
    for j in range(n):
        obj_expr += cost_quad[i][j] * y[i, j] + cost_lin[i][j] * x[i, j]

model.setObjective(obj_expr, GRB.MINIMIZE)

# =========================
# 6. Constraints
# =========================

# 6.1 Customer demand constraints: sum_i x[i,j] = d[j]
for j in range(n):
    model.addConstr(
        gp.quicksum(x[i, j] for i in range(m)) == d[j],
        name=f"Demand_Customer{j+1}"
    )

# 6.2 Warehouse supply constraints: sum_j x[i,j] <= s_max[i]
for i in range(m):
    model.addConstr(
        gp.quicksum(x[i, j] for j in range(n)) <= s_max[i],
        name=f"Supply_Warehouse{i+1}"
    )

# Nonnegativity already enforced via lb=0 on x-variables

# =========================
# 7. Solve model
# =========================
model.optimize()

# =========================
# 8. Print results
# =========================
if model.Status == GRB.OPTIMAL:
    print("Optimal solution found.")
    print(f"Minimum total transportation cost: {model.ObjVal:.6f}")
    for i in range(m):
        for j in range(n):
            print(f"x[{i+1},{j+1}] = {x[i, j].X:.6f}")
else:
    print(f"Optimization ended with status {model.Status}")

# =========================
# 9. FinalAnswer output (minimum transportation cost)
# =========================
if model.Status == GRB.OPTIMAL:
    final_answer = model.ObjVal
else:
    final_answer = float('nan')

print(f"FinalAnswer=【{final_answer}】")