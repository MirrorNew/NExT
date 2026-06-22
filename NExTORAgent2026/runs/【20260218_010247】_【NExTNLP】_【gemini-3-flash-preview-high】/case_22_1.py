import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
# Parameters extracted from the provided list
total_vehicles = 100
T1_constant_term = 10.0
T1_quadratic_coefficient = 0.1
T2_constant_term = 12.0
T2_quadratic_coefficient = 0.05
actual_travel_time_exponent = 1.05

# 2. Initialize the Gurobi model.
model = gp.Model("RoadNetworkOptimization")

# 3. Create decision variables.
# From context: x is number of vehicles on route 1, y is number of vehicles on route 2
x = model.addVar(lb=0, ub=total_vehicles, vtype=GRB.INTEGER, name="x")
y = model.addVar(lb=0, ub=total_vehicles, vtype=GRB.INTEGER, name="y")

# 4. Create auxiliary substitution variables.
# As per instructions, auxiliary substitution variables range from negative infinity to positive infinity.
x_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x_sq")
y_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y_sq")
T1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="T1")
T2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="T2")
T1_act = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="T1_act")
T2_act = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="T2_act")
Z1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Z1")
Z2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Z2")

# 5. Set up the objective function.
# Minimize total actual travel time Z = Z1 + Z2
model.setObjective(Z1 + Z2, GRB.MINIMIZE)

# 6. Add all constraints (including gen-constr and indicators).
# Set NonConvex parameter for general power constraints and bilinear products
model.Params.NonConvex = 2

# Vehicle total distribution constraint
model.addConstr(x + y == total_vehicles, name="VehicleDistribution")

# Square the vehicle counts for the expected time formulas
# Y = X^2 requires model.addGenConstrPow(X,Y,2)
model.addGenConstrPow(x, x_sq, 2.0)
model.addGenConstrPow(y, y_sq, 2.0)

# Expected travel time definitions
model.addConstr(T1 == T1_constant_term + T1_quadratic_coefficient * x_sq, name="T1_definition")
model.addConstr(T2 == T2_constant_term + T2_quadratic_coefficient * y_sq, name="T2_definition")

# Actual travel time as expected travel time to the power of 1.05
# Y = X^a requires model.addGenConstrPow(X,Y,a)
model.addGenConstrPow(T1, T1_act, actual_travel_time_exponent)
model.addGenConstrPow(T2, T2_act, actual_travel_time_exponent)

# Total actual travel time contribution per route (Bilinear products)
# Using direct multiplication instead of addGenConstrMul as per instructions
model.addConstr(Z1 == x * T1_act, name="Z1_definition")
model.addConstr(Z2 == y * T2_act, name="Z2_definition")

# 7. Solve the model and print results.
model.optimize()

if model.Status == GRB.OPTIMAL:
    # Extract the value of vehicles on route 1
    vehicle_count_route_1 = x.X
    print(f"FinalAnswer=【{int(round(vehicle_count_route_1))}】")