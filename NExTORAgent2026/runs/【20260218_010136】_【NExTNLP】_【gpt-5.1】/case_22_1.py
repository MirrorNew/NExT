import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Define all parameters
# =========================
num_parallel_main_roads = 2              # not directly used in model, kept for completeness
total_commuter_vehicles = 100000         # background parameter
predicted_journey_time = 35              # background parameter (minutes)
actual_time_increase = 0.07              # background parameter
upgrade_year = 2024                      # background parameter
travel_time_reduction = 0.12             # background parameter
difference_from_theoretical = 0.02       # background parameter
num_parallel_roads = 2                   # there are two roads in the current problem

total_vehicles = 100                     # total vehicles between the two routes

T1_constant_term = 10.0                  # in minutes
T1_quadratic_coefficient = 0.1          # coefficient in T1 = 10 + 0.1 * x^2

T2_constant_term = 12.0                  # in minutes
T2_quadratic_coefficient = 0.05         # coefficient in T2 = 12 + 0.05 * y^2

actual_travel_time_exponent = 1.05      # T_act = T^1.05

# =========================
# 2. Create model
# =========================
model = gp.Model("Kazidong_Road_Traffic_Assignment")

# Enable nonconvex general constraints (for power functions with exponent 1.05)
model.Params.NonConvex = 2

# =========================
# 3. Decision variables
# =========================
# x: number of vehicles on route 1
x = model.addVar(vtype=GRB.INTEGER, lb=0, ub=total_vehicles, name="x")

# y: number of vehicles on route 2
y = model.addVar(vtype=GRB.INTEGER, lb=0, ub=total_vehicles, name="y")

# T1, T2: expected travel times per vehicle on each route
T1 = model.addVar(vtype=GRB.CONTINUOUS, lb=T1_constant_term, name="T1")
T2 = model.addVar(vtype=GRB.CONTINUOUS, lb=T2_constant_term, name="T2")

# T1_act, T2_act: actual travel times per vehicle on each route
T1_act = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="T1_act")
T2_act = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="T2_act")

# =========================
# 4. Auxiliary variables
# =========================
# q1, q2 for squares of flows
q1 = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="q1")  # q1 = x^2
q2 = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="q2")  # q2 = y^2

# t1pow, t2pow for 1.05 powers of expected times
t1pow = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t1pow")  # t1pow = T1^1.05
t2pow = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t2pow")  # t2pow = T2^1.05

# =========================
# 5. Objective function
# =========================
# Minimize total actual travel time: Z = x*T1_act + y*T2_act
model.setObjective(x * T1_act + y * T2_act, GRB.MINIMIZE)

# =========================
# 6. Constraints
# =========================

# 6.1 Vehicle distribution (conservation)
model.addConstr(x + y == total_vehicles, name="vehicle_conservation")

# 6.2 Expected travel time definitions using auxiliary quadratic variables
# q1 = x^2
model.addGenConstrPow(x, q1, 2.0, name="q1_def")
# T1 = 10 + 0.1 * q1
model.addConstr(T1 == T1_constant_term + T1_quadratic_coefficient * q1, name="T1_def")

# q2 = y^2
model.addGenConstrPow(y, q2, 2.0, name="q2_def")
# T2 = 12 + 0.05 * q2
model.addConstr(T2 == T2_constant_term + T2_quadratic_coefficient * q2, name="T2_def")

# 6.3 Actual travel time definitions using power 1.05
# t1pow = T1^1.05
model.addGenConstrPow(T1, t1pow, actual_travel_time_exponent, name="t1pow_def")
model.addConstr(T1_act == t1pow, name="T1_act_def")

# t2pow = T2^1.05
model.addGenConstrPow(T2, t2pow, actual_travel_time_exponent, name="t2pow_def")
model.addConstr(T2_act == t2pow, name="T2_act_def")

# 6.4 Nonnegativity already ensured by lb in variable creation; no denominator constraints or indicators needed.

# =========================
# 7. Optimize model
# =========================
model.optimize()

# =========================
# 8. Print results
# =========================
if model.Status == GRB.OPTIMAL:
    x_opt = int(round(x.X))
    y_opt = int(round(y.X))
    T1_opt = T1.X
    T2_opt = T2.X
    T1_act_opt = T1_act.X
    T2_act_opt = T2_act.X
    total_actual_time = x_opt * T1_act_opt + y_opt * T2_act_opt

    print("Optimal solution found:")
    print(f"  Vehicles on route 1 (x): {x_opt}")
    print(f"  Vehicles on route 2 (y): {y_opt}")
    print(f"  Expected time route 1 (T1): {T1_opt:.6f} minutes")
    print(f"  Expected time route 2 (T2): {T2_opt:.6f} minutes")
    print(f"  Actual time route 1 (T1_act): {T1_act_opt:.6f} minutes")
    print(f"  Actual time route 2 (T2_act): {T2_act_opt:.6f} minutes")
    print(f"  Total actual travel time: {total_actual_time:.6f} vehicle-minutes")

    # The question asks: determine the value of the number of vehicles on route 1.
    # So FinalAnswer is x_opt.
    print(f"FinalAnswer=【{x_opt}】")
else:
    print(f"Optimization ended with status {model.Status}")
    # In a non-optimal case, we still need to output FinalAnswer; we set it to None.
    print("FinalAnswer=【None】")