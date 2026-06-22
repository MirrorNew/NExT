import gurobipy as gp

# Model
model = gp.Model("ParallelRoadMinimizeTotalTravelTime")

# Parameters (using provided parameters list)
total_vehicles = 100
T1_constant_term = 10.0
T1_quadratic_coefficient = 0.1
T2_constant_term = 12.0
T2_quadratic_coefficient = 0.05
actual_travel_time_exponent = 1.05

# Decision variables
x = model.addVar(lb=0, ub=total_vehicles, vtype=gp.GRB.INTEGER, name="x")
y = model.addVar(lb=0, ub=total_vehicles, vtype=gp.GRB.INTEGER, name="y")

# Auxiliary substitution variables for squares
X2 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="X2")
Y2 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="Y2")

# Expected travel time variables
T1 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="T1")
T2 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="T2")

# Auxiliary substitution variables for powers
T1_pow = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="T1_pow")
T2_pow = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="T2_pow")

# Actual travel time variables
T1_act = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="T1_act")
T2_act = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="T2_act")

# Set nonconvex parameter
model.Params.NonConvex = 2

# Constraints
# Vehicle distribution
model.addConstr(x + y == total_vehicles, name="total_vehicles")

# Define x² and y² using general constraints
model.addGenConstrPow(x, X2, 2, name="X2_is_x_squared")
model.addGenConstrPow(y, Y2, 2, name="Y2_is_y_squared")

# Expected travel time definitions
model.addConstr(T1 == T1_constant_term + T1_quadratic_coefficient * X2, name="T1_def")
model.addConstr(T2 == T2_constant_term + T2_quadratic_coefficient * Y2, name="T2_def")

# Actual travel time definitions (power functions)
model.addGenConstrPow(T1, T1_pow, actual_travel_time_exponent, name="T1_power")
model.addGenConstrPow(T2, T2_pow, actual_travel_time_exponent, name="T2_power")

# Link power variables to actual travel time variables
model.addConstr(T1_act == T1_pow, name="T1_act_eq")
model.addConstr(T2_act == T2_pow, name="T2_act_eq")

# Objective: minimize total actual travel time
model.setObjective(x * T1_act + y * T2_act, gp.GRB.MINIMIZE)

# Solve
model.optimize()

# Output results
if model.status == gp.GRB.OPTIMAL:
    x_val = int(round(x.x))
    y_val = int(round(y.x))
    T1_val = T1.x
    T2_val = T2.x
    T1_act_val = T1_act.x
    T2_act_val = T2_act.x
    obj_val = model.objVal
    
    print(f"Optimal number of vehicles on route 1 (x) = {x_val}")
    print(f"Number of vehicles on route 2 (y) = {y_val}")
    print(f"Expected travel time per vehicle on route 1 = {T1_val:.2f} minutes")
    print(f"Expected travel time per vehicle on route 2 = {T2_val:.2f} minutes")
    print(f"Actual travel time per vehicle on route 1 = {T1_act_val:.2f} minutes")
    print(f"Actual travel time per vehicle on route 2 = {T2_act_val:.2f} minutes")
    print(f"Total actual travel time = {obj_val:.2f} minutes")
    
    # Final answer
    print(f"FinalAnswer=【{x_val}】")
else:
    print("No optimal solution found.")
    print(f"FinalAnswer=【0】")