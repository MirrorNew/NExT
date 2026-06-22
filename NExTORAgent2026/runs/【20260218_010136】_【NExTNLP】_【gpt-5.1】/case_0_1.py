import gurobipy as gp
from gurobipy import GRB

# ==============================
# 1. Parameters (must use given list)
# ==============================
number_of_sections = 4
number_of_work_sites = 4
Q = [10, 15, 20, 25]
coordinates = [
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2]
]

# Extract x_i, y_i from coordinates
x_i = [coordinates[i][0] for i in range(number_of_work_sites)]
y_i = [coordinates[i][1] for i in range(number_of_work_sites)]

# ==============================
# 2. Create model
# ==============================
model = gp.Model("Concrete_Mixing_Station_Location")

# Allow non-convex quadratic / general constraint modeling
model.Params.NonConvex = 2

# ==============================
# 3. Decision variables
# ==============================
# Coordinates of the central mixing station
x = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
y = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y")

# Distance from station to each work site (d_i >= 0)
d = model.addVars(number_of_work_sites, lb=0.0, ub=GRB.INFINITY,
                  vtype=GRB.CONTINUOUS, name="d")

# ==============================
# 4. Auxiliary substitution variables
# ==============================
# dx_i = x - x_i, dy_i = y - y_i
dx = model.addVars(number_of_work_sites, lb=-GRB.INFINITY, ub=GRB.INFINITY,
                   vtype=GRB.CONTINUOUS, name="dx")
dy = model.addVars(number_of_work_sites, lb=-GRB.INFINITY, ub=GRB.INFINITY,
                   vtype=GRB.CONTINUOUS, name="dy")

# sqdx_i = (dx_i)^2, sqdy_i = (dy_i)^2
sqdx = model.addVars(number_of_work_sites, lb=-GRB.INFINITY, ub=GRB.INFINITY,
                     vtype=GRB.CONTINUOUS, name="sqdx")
sqdy = model.addVars(number_of_work_sites, lb=-GRB.INFINITY, ub=GRB.INFINITY,
                     vtype=GRB.CONTINUOUS, name="sqdy")

# d2_i = (d_i)^2
d2 = model.addVars(number_of_work_sites, lb=-GRB.INFINITY, ub=GRB.INFINITY,
                   vtype=GRB.CONTINUOUS, name="d2")

# ==============================
# 5. Objective function
#   minimize sum_i Q_i * d_i
# ==============================
model.setObjective(gp.quicksum(Q[i] * d[i] for i in range(number_of_work_sites)),
                   GRB.MINIMIZE)

# ==============================
# 6. Constraints
# ==============================

for i in range(number_of_work_sites):
    # Linear relations for dx_i and dy_i
    model.addConstr(dx[i] == x - x_i[i], name=f"dx_def_{i}")
    model.addConstr(dy[i] == y - y_i[i], name=f"dy_def_{i}")

    # Squares: sqdx_i = (dx_i)^2, sqdy_i = (dy_i)^2
    model.addGenConstrPow(dx[i], sqdx[i], 2.0, name=f"sqdx_def_{i}")
    model.addGenConstrPow(dy[i], sqdy[i], 2.0, name=f"sqdy_def_{i}")

    # d2_i = d_i^2
    model.addGenConstrPow(d[i], d2[i], 2.0, name=f"d2_def_{i}")

    # Distance definition (without sqrt):
    # d_i^2 = (x - x_i)^2 + (y - y_i)^2
    model.addConstr(d2[i] == sqdx[i] + sqdy[i], name=f"dist_eq_{i}")

# ==============================
# 7. Solve model and print results
# ==============================
model.optimize()

if model.Status == GRB.OPTIMAL:
    x_star = x.X
    y_star = y.X
    obj_val = model.ObjVal

    print("Optimal solution found.")
    print(f"Location of mixing station: x = {x_star:.6f}, y = {y_star:.6f}")
    print(f"Minimum total transportation volume (sum Q_i * d_i) = {obj_val:.6f}")

    # As an example, FinalAnswer is chosen as the optimal objective value
    FinalAnswer = obj_val
else:
    print(f"Optimization ended with status {model.Status}")
    FinalAnswer = float('nan')

# Required final output statement
print(f"FinalAnswer=【{FinalAnswer}】")