import gurobipy as gp
from gurobipy import GRB

# 1. Define parameters based on the provided Parameters List
total_vehicles = 100
T1_constant_term = 10.0
T1_quadratic_coefficient = 0.1
T2_constant_term = 12.0
T2_quadratic_coefficient = 0.05
actual_travel_time_exponent = 1.05

# 2. Create the model
model = gp.Model("Traffic_Optimization")

# Set NonConvex parameter to 2 to handle non-convex quadratic and general constraints
model.Params.NonConvex = 2

# 3. Create decision variables
# x: Number of vehicles taking route 1
x = model.addVar(lb=0, ub=total_vehicles, vtype=GRB.INTEGER, name="x")
# y: Number of vehicles taking route 2
y = model.addVar(lb=0, ub=total_vehicles, vtype=GRB.INTEGER, name="y")

# 4. Create auxiliary substitution variables
# Variables for quadratic terms
x_sq = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x_sq")
y_sq = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y_sq")

# Variables for expected travel times
T1 = model.addVar(lb=10, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="T1")
T2 = model.addVar(lb=12, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="T2")

# Variables for actual travel times (power 1.05)
T1_act = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="T1_act")
T2_act = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="T2_act")

# Variables for total travel time components
Z1 = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Z1")
Z2 = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Z2")

# 5. Set up the objective function
# Minimize total actual travel time Z = Z1 + Z2
model.setObjective(Z1 + Z2, GRB.MINIMIZE)

# 6. Add constraints

# Vehicle distribution constraint
model.addConstr(x + y == total_vehicles, name="Distribution")

# Quadratic definitions: x_sq = x^2, y_sq = y^2
model.addGenConstrPow(x, x_sq, 2, name="Def_x_sq")
model.addGenConstrPow(y, y_sq, 2, name="Def_y_sq")

# Expected travel time definitions
# T1 = 10 + 0.1 * x^2
model.addConstr(T1 == T1_constant_term + T1_quadratic_coefficient * x_sq, name="Def_T1")
# T2 = 12 + 0.05 * y^2
model.addConstr(T2 == T2_constant_term + T2_quadratic_coefficient * y_sq, name="Def_T2")

# Actual travel time definitions (Power constraints)
# T1_act = T1 ^ 1.05
model.addGenConstrPow(T1, T1_act, actual_travel_time_exponent, name="Def_T1_act")
# T2_act = T2 ^ 1.05
model.addGenConstrPow(T2, T2_act, actual_travel_time_exponent, name="Def_T2_act")

# Total time per route definitions (Bilinear constraints)
# Z1 = x * T1_act
model.addConstr(Z1 == x * T1_act, name="Def_Z1")
# Z2 = y * T2_act
model.addConstr(Z2 == y * T2_act, name="Def_Z2")

# 7. Solve the model and print results
model.optimize()

if model.Status == GRB.OPTIMAL:
    print(f"Optimal objective value: {model.ObjVal}")
    print(f"x (Route 1 vehicles): {x.X}")
    print(f"y (Route 2 vehicles): {y.X}")
    
    # The question asks for the number of vehicles on route 1
    print(f"FinalAnswer=【{int(round(x.X))}】")
else:
    print("Optimization was not successful.")