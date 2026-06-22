import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Import parameters (from Parameters List)
# =========================
E = [100, 150, 80]          # Max feasible emission reductions for each factory (tons)
alpha = [0.5, 0.8, 1.0]     # Cost coefficients (10,000 yuan per (ton^2))
R_min = 120                 # Minimum total emission reduction (tons)
bonus_rate = 60             # Bonus rate (10,000 yuan per excess ton)

n = len(E)

# =========================
# 2. Create model
# =========================
model = gp.Model("Kazdale_Emission_Reduction")

# Allow quadratic / nonconvex constructs
model.Params.NonConvex = 2

# =========================
# 3. Decision variables
# =========================
# x[i]: emission reduction of factory i (tons), 0 <= x[i] <= E[i]
x = model.addVars(n, lb=0.0, name="x")
for i in range(n):
    x[i].UB = E[i]

# B: total bonus (10,000 yuan), B >= 0
B = model.addVar(lb=0.0, name="B")

# =========================
# 4. Auxiliary substitution variables
# =========================
# y[i] = x[i]^2, used to linearize the quadratic term in objective
y = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y")

# Power constraints: y[i] = x[i]^2
for i in range(n):
    model.addGenConstrPow(x[i], y[i], 2.0, name=f"Pow_x2_{i+1}")

# =========================
# 5. Objective function
# Minimize total cost: sum(alpha[i] * x[i]^2) - B
# Implemented via y[i] = x[i]^2
# =========================
objective_expr = gp.quicksum(alpha[i] * y[i] for i in range(n)) - B
model.setObjective(objective_expr, GRB.MINIMIZE)

# =========================
# 6. Constraints
# =========================

# (a) Emission reduction bounds already encoded in variable lb and ub:
#     0 <= x[i] <= E[i], i = 1,2,3

# (b) Total reduction requirement: sum x[i] >= R_min
total_reduction = gp.quicksum(x[i] for i in range(n))
model.addConstr(total_reduction >= R_min, name="TotalReduction")

# (c) Bonus definition:
#     B >= bonus_rate * (sum x[i] - R_min)
#     B >= 0 (already via lb, we still add explicit constraint as in context)
model.addConstr(B >= 0, name="B_nonnegative")
model.addConstr(B >= bonus_rate * (total_reduction - R_min),
                name="B_lower_bound")

# (d) Bonus–reduction linkage (upper bound):
#     B <= bonus_rate * (sum x[i] - R_min)
model.addConstr(B <= bonus_rate * (total_reduction - R_min),
                name="B_upper_bound")

# =========================
# 7. Solve the model
# =========================
model.optimize()

# =========================
# 8. Print results and FinalAnswer
# =========================
if model.Status == GRB.OPTIMAL:
    x_vals = [x[i].X for i in range(n)]
    B_val = B.X
    total_reduction_val = sum(x_vals)
    objective_val = model.ObjVal

    print("Optimal solution found:")
    for i in range(n):
        print(f"Factory {i+1} emission reduction x_{i+1} = {x_vals[i]:.4f} tons")
    print(f"Total reduction = {total_reduction_val:.4f} tons")
    print(f"Bonus B = {B_val:.4f} (10,000 yuan)")
    print(f"Objective value (total cost after bonus) = {objective_val:.4f} (10,000 yuan)")

    # The question asks: "How should the emission reduction of each factory be planned?"
    # FinalAnswer is the vector of emission reductions [x1, x2, x3].
    final_answer_str = "[" + ", ".join(f"{val:.4f}" for val in x_vals) + "]"
    print(f"FinalAnswer=【{final_answer_str}】")
else:
    # If no optimal solution, output according to required format
    print("No optimal solution found.")
    print("FinalAnswer=【No feasible optimal solution】")