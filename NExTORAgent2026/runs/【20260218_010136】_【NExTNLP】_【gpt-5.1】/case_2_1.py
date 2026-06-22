import gurobipy as gp
from gurobipy import GRB
import math

# 1. Define parameters (from Parameters List)
r = 300  # radius of the circular steel billet

# We will use the reduced model:
# max  M_max = (4/3) * sigma_y * x1 * x2^2
# s.t. x1^2 + x2^2 <= r^2
#      x1 >= 0, x2 >= 0
#
# Since sigma_y is a positive constant multiplier, maximizing x1*x2^2 is equivalent.
# To stay consistent with the statement "You must strictly use the Value in the Parameters List",
# we treat sigma_y as a positive constant = 1 (scaling does not affect x1*), and focus on maximizing x1*x2^2.

sigma_y = 1.0  # positive constant; does not affect the optimizer x1*, only scales objective

# 2. Create model
model = gp.Model("BeamSectionOptimization")

# Allow non-convex quadratic constraints
model.Params.NonConvex = 2

# 3. Create decision variables
x1 = model.addVar(lb=0.0, ub=r, vtype=GRB.CONTINUOUS, name="x1")  # half-width
x2 = model.addVar(lb=0.0, ub=r, vtype=GRB.CONTINUOUS, name="x2")  # half-height

# 4. Create auxiliary substitution variables
# t1 = x1^2
t1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="t1")
# t2 = x2^2
t2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="t2")
# t3 = x1 * x2^2 = x1 * t2 (represents the geometric term to maximize)
t3 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="t3")

# 5. Set up the objective function
# M_max = (4/3) * sigma_y * x1 * x2^2 = (4/3) * sigma_y * t3
M_max_expr = (4.0 / 3.0) * sigma_y * t3
model.setObjective(M_max_expr, GRB.MAXIMIZE)

# 6. Add constraints

# Nonlinear definitions via general constraints
# t1 = x1^2
model.addGenConstrPow(x1, t1, 2.0, name="def_t1")
# t2 = x2^2
model.addGenConstrPow(x2, t2, 2.0, name="def_t2")
# t3 = x1 * t2
model.addConstr(x1 * t2 == t3, name="def_t3")

# Circular-billet boundary: x1^2 + x2^2 <= r^2  --> t1 + t2 <= r^2
model.addConstr(t1 + t2 <= r * r, name="circular_boundary")

# Non-negativity of x1, x2 already handled by variable bounds

# 7. Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    x1_opt = x1.X
    x2_opt = x2.X
    t3_opt = t3.X
    M_max_opt = M_max_expr.getValue()

    print("Optimal solution found:")
    print(f"x1 (half-width)  = {x1_opt:.6f}")
    print(f"x2 (half-height) = {x2_opt:.6f}")
    print(f"x1^2 + x2^2      = {t1.X + t2.X:.6f} (should be <= {r*r})")
    print(f"x1 * x2^2 (t3)   = {t3_opt:.6f}")
    print(f"Objective (M_max) = {M_max_opt:.6f} (scaled by sigma_y={sigma_y})")

    # According to analytic result: x1* = r / sqrt(3)
    x1_analytic = r / math.sqrt(3.0)
    print(f"Analytic x1* (r/sqrt(3)) ≈ {x1_analytic:.6f}")

    # Final answer for the question: value of x1
    print(f"FinalAnswer=【{x1_opt}】")
else:
    print("No optimal solution found.")
    # In case of no optimal solution, still print something for FinalAnswer
    print("FinalAnswer=【nan】")