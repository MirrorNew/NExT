import gurobipy as gp
from gurobipy import GRB

# ===============================
# 1. Define parameters (from Parameters List)
# ===============================
budget_upper_limit = 20                 # million USD
total_investment_bound = 20             # million USD (same as budget limit here)
A = [50, 30]                            # coefficients A for the two projects
k = [0.3, 0.6]                          # coefficients k for the two projects
peak_sales_project1 = 5.0               # not directly used in model
market_space_project2 = 3.0             # not directly used in model
year = 2023                             # not directly used in model

# ===============================
# 2. Create model
# ===============================
model = gp.Model("R&D_Investment_Allocation")

# Allow non-convex general constraints (exp is handled through gen-constr-exponential)
model.Params.NonConvex = 2

# ===============================
# 3. Decision variables
# ===============================
# x1: investment in Project 1 (anti-tumor), in million USD
# x2: investment in Project 2 (vaccine improvement), in million USD
x1 = model.addVar(lb=0.0, ub=total_investment_bound, vtype=GRB.CONTINUOUS, name="x1")
x2 = model.addVar(lb=0.0, ub=total_investment_bound, vtype=GRB.CONTINUOUS, name="x2")

# ===============================
# 4. Auxiliary substitution variables
# ===============================
# u1 = exp(-0.3 * x1)
# u2 = exp(-0.6 * x2)
# Z is the total return
u1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="u1")
u2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="u2")
Z = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Z")

# Exponential general constraints:
# Gurobi form: addGenConstrExp(x, y) enforces y = exp(x)
# We need u1 = exp(-0.3 * x1), u2 = exp(-0.6 * x2)
t1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="t1")
t2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="t2")

# Linear definitions of t1 and t2
model.addConstr(t1 == -k[0] * x1, name="def_t1")  # t1 = -0.3 * x1
model.addConstr(t2 == -k[1] * x2, name="def_t2")  # t2 = -0.6 * x2

# Exponential constraints: u1 = exp(t1), u2 = exp(t2)
model.addGenConstrExp(t1, u1, name="exp1")
model.addGenConstrExp(t2, u2, name="exp2")

# Link Z to u1 and u2:
# Z = 50 * (1 - u1) + 30 * (1 - u2)
model.addConstr(
    Z == A[0] * (1 - u1) + A[1] * (1 - u2),
    name="def_Z"
)

# ===============================
# 5. Objective function
# ===============================
# Maximize total return Z
model.setObjective(Z, GRB.MAXIMIZE)

# ===============================
# 6. Constraints
# ===============================
# Budget constraint: x1 + x2 <= 20
model.addConstr(x1 + x2 <= budget_upper_limit, name="budget")

# Non-negativity already enforced via lb=0; add explicit constraints if desired:
model.addConstr(x1 >= 0, name="nonneg_x1")
model.addConstr(x2 >= 0, name="nonneg_x2")

# ===============================
# 7. Solve the model
# ===============================
model.optimize()

# ===============================
# 8. Print results
# ===============================
if model.status == GRB.OPTIMAL:
    opt_x1 = x1.X
    opt_x2 = x2.X
    opt_Z = Z.X

    print(f"Optimal solution status: {model.Status}")
    print(f"Optimal investment in Project 1 (x1): {opt_x1:.6f} million USD")
    print(f"Optimal investment in Project 2 (x2): {opt_x2:.6f} million USD")
    print(f"Maximum total return Z: {opt_Z:.6f} million USD")

    # For completeness, print u1 and u2
    print(f"u1 = exp(-0.3 * x1) ≈ {u1.X:.6f}")
    print(f"u2 = exp(-0.6 * x2) ≈ {u2.X:.6f}")
else:
    print(f"Optimization ended with status {model.Status}")
    opt_Z = float('nan')

# ===============================
# 9. Final answer output
# ===============================
# According to the problem statement, the "question answer" is the best (maximum) total return.
print(f"FinalAnswer=【{opt_Z}】")