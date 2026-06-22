import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and any other necessary packages.
# (Done above)

# 2. Define all parameter matrices and data inputs.
# From the problem description and parameters list
r = 300
# Yield limit of the steel sigma_y is mentioned but not given a specific value.
# Since the optimal geometric configuration (x1, x2) that maximizes the moment M 
# is independent of the specific positive value of sigma_y, we set sigma_y = 1.0.
sigma_y = 1.0

# 3. Create a new Gurobi model.
model = gp.Model("BridgeBeamOptimization")

# 4. Create decision variables.
# x1: half-width of the rectangular cross-section
# x2: half-height of the rectangular cross-section
# M: bending moment the beam can withstand
# sigma_max: maximum bending stress on the beam section
x1 = model.addVar(lb=0, ub=r, vtype=GRB.CONTINUOUS, name="x1")
x2 = model.addVar(lb=0, ub=r, vtype=GRB.CONTINUOUS, name="x2")
M = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="M")
sigma_max = model.addVar(lb=0, ub=sigma_y, vtype=GRB.CONTINUOUS, name="sigma_max")

# 5. Create auxiliary substitution variables in coding advice.
# (The values range from negative infinity to positive infinity, lb=-GRB.INFINITY, ub=GRB.INFINITY).
x1_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x1_sq")
x2_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x2_sq")
x1_x2_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x1_x2_sq")

# 6. Set up the objective function.
# The goal is to maximize the bending moment M that the beam can withstand.
model.setObjective(M, GRB.MAXIMIZE)

# 7. Add all constraints.
# Circular-billet boundary: the rectangle must fit inside the circular stick
# x1^2 + x2^2 <= r^2
model.addGenConstrPow(x1, x1_sq, 2)
model.addGenConstrPow(x2, x2_sq, 2)
model.addConstr(x1_sq + x2_sq <= r**2, name="circular_boundary")

# Substitution for the cubic term x1 * x2^2
model.addConstr(x1 * x2_sq == x1_x2_sq, name="cubic_term_substitution")

# Maximum stress definition: sigma_max = (3/4) * M / (x1 * x2^2)
# Rewriting to avoid variables in the denominator: sigma_max * x1_x2_sq == 0.75 * M
model.addConstr(sigma_max * x1_x2_sq == 0.75 * M, name="stress_definition")

# Yield-stress limit: sigma_max <= sigma_y
# (Already enforced by the upper bound of the sigma_max variable)

# Enforce the yield limit on the bending moment M based on structural mechanics:
# M <= (4/3) * sigma_y * x1 * x2^2
model.addConstr(M <= (4.0 / 3.0) * sigma_y * x1_x2_sq, name="yield_limit_on_moment")

# Set Gurobi parameter for Non-Convex optimization as the model contains products of variables.
model.Params.NonConvex = 2

# 8. Solve the model and print results.
model.optimize()

if model.status == GRB.OPTIMAL:
    the_question_answer = x1.X
    print(f"FinalAnswer=【{the_question_answer}】")
else:
    print("FinalAnswer=【No solution found】")