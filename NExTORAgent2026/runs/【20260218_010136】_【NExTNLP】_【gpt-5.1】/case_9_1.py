import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Define all parameters
# =========================

number_of_companies = 12
shift_count = 3
number_of_plants = 12
I = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
x_i = [1.0, 3.0, 0.5, 5.0, 2.0, 4.0, 1.5, 3.5, 0.8, 2.5, 4.5, 1.2]
y_i = [2.0, 1.5, 4.0, 3.0, 2.5, 5.0, 1.0, 4.5, 3.2, 0.5, 2.0, 5.5]
p_i = [50, 80, 30, 100, 60, 70, 40, 90, 55, 75, 85, 65]
safety_distance = 0.8
distance_metric_power = 1
Table_1_FactoryData = [
    [1, 1.0, 2.0, 50],
    [2, 3.0, 1.5, 80],
    [3, 0.5, 4.0, 30],
    [4, 5.0, 3.0, 100],
    [5, 2.0, 2.5, 60],
    [6, 4.0, 5.0, 70],
    [7, 1.5, 1.0, 40],
    [8, 3.5, 4.5, 90],
    [9, 0.8, 3.2, 55],
    [10, 2.5, 0.5, 75],
    [11, 4.5, 2.0, 85],
    [12, 1.2, 5.5, 65],
]

# Index mapping i -> 0-based index k
idx_map = {i: i - 1 for i in I}

# ====================================================
# 2. Create model (Euclidean distance, nonlinear NLP)
# ====================================================

model = gp.Model("Centralized_Hazardous_Warehouse_Location")

# Nonconvex quadratic and sqrt-type modeling will be used
model.Params.NonConvex = 2

# =========================
# 3. Decision variables
# =========================

# Warehouse coordinates
x = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
y = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y")

# ================================================
# 4. Auxiliary substitution / distance variables
# ================================================

# d_i: Euclidean distance from (x,y) to plant i (nonnegative)
d = model.addVars(I, lb=0.0, vtype=GRB.CONTINUOUS, name="d")

# t_i: squared distance from (x,y) to plant i = d_i^2 (can be any real, but here nonnegative)
t = model.addVars(I, lb=0.0, vtype=GRB.CONTINUOUS, name="t")

# (We keep bounds on t >= 0; requirement says auxiliary vars range can be -inf..inf,
# but nonnegativity here is valid and tighter, and does not conflict with any denominator/log/pow handling.)

# ==================================
# 5. Objective function
# ==================================

# minimize sum_{i} p_i * d_i
model.setObjective(
    gp.quicksum(p_i[idx_map[i]] * d[i] for i in I),
    GRB.MINIMIZE
)

# =========================
# 6. Constraints
# =========================

# 6.1 Define squared distances t_i = (x - x_i)^2 + (y - y_i)^2
for i in I:
    k = idx_map[i]
    model.addConstr(
        t[i] == (x - x_i[k]) * (x - x_i[k]) + (y - y_i[k]) * (y - y_i[k]),
        name=f"squared_distance_def_{i}"
    )

# 6.2 Link d_i and t_i via power constraint: t_i = d_i^2
for i in I:
    # d[i] is the "X" variable, t[i] is the "Y" variable in Y = X^a
    model.addGenConstrPow(d[i], t[i], 2.0, name=f"d_sq_def_{i}")

# 6.3 Safety distance constraints: sqrt((x-x_i)^2 + (y-y_i)^2) >= safety_distance
# Using t_i = (x-x_i)^2 + (y-y_i)^2:  t_i >= safety_distance^2
for i in I:
    model.addConstr(
        t[i] >= safety_distance * safety_distance,
        name=f"safety_distance_{i}"
    )

# ==================================
# 7. Solve the model and print results
# ==================================

model.optimize()

if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT or model.Status == GRB.SUBOPTIMAL:
    x_val = x.X
    y_val = y.X
    obj_val = model.ObjVal

    print("Optimal solution (or best found):")
    print(f"  Warehouse location x = {x_val:.6f}")
    print(f"  Warehouse location y = {y_val:.6f}")
    print(f"  Objective (weighted total distance) = {obj_val:.6f}")

    print("\nDistances to each plant:")
    for i in I:
        print(f"  Plant {i}: distance d[{i}] = {d[i].X:.6f}")

    # According to the problem statement, the "answer to the question"
    # is the optimal warehouse location (x, y). We output that in the required format.
    final_answer = (x_val, y_val)
    print(f"FinalAnswer=【{final_answer}】")
else:
    print("No optimal solution found.")
    # Still print in required format, but indicate infeasibility or no solution
    print("FinalAnswer=【No feasible solution】")