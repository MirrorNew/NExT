import gurobipy as gp

# ========== Parameters from Parameters List ==========
total_vehicles = 2100
capacity_A = 1000
capacity_B = 1200
threshold_A = 900  # flow_threshold_A = 900 (0.9 * 1000)
threshold_B = 1080  # 0.9 * 1200
idle_time_A = 10
idle_time_B = 12
coef_A = 0.0001
coef_B = 0.00008
additional_rate = 0.05

# ========== Create Gurobi Model ==========
model = gp.Model("TrafficAllocation")

# ========== Decision Variables ==========
f_A = model.addVar(lb=0, ub=capacity_A, name="f_A")
f_B = model.addVar(lb=0, ub=capacity_B, name="f_B")
T_A = model.addVar(lb=idle_time_A, ub=gp.GRB.INFINITY, name="T_A")
T_B = model.addVar(lb=idle_time_B, ub=gp.GRB.INFINITY, name="T_B")

# ========== Auxiliary Variables for Quadratic Terms ==========
fA_sq = model.addVar(lb=0, ub=gp.GRB.INFINITY, name="fA_sq")
fB_sq = model.addVar(lb=0, ub=gp.GRB.INFINITY, name="fB_sq")

# ========== Auxiliary Variables for Base and Delayed Travel Times ==========
base_TA = model.addVar(lb=idle_time_A, ub=gp.GRB.INFINITY, name="base_TA")
base_TB = model.addVar(lb=idle_time_B, ub=gp.GRB.INFINITY, name="base_TB")
delay_TA = model.addVar(lb=idle_time_A, ub=gp.GRB.INFINITY, name="delay_TA")
delay_TB = model.addVar(lb=idle_time_B, ub=gp.GRB.INFINITY, name="delay_TB")

# ========== Binary Indicator Variables ==========
y_A = model.addVar(vtype=gp.GRB.BINARY, name="y_A")
y_B = model.addVar(vtype=gp.GRB.BINARY, name="y_B")

# ========== Enable Non-Convex Processing ==========
model.Params.NonConvex = 2

# ========== Constraints ==========
# Flow conservation
model.addConstr(f_A + f_B == total_vehicles, name="flow_conservation")

# Quadratic terms via general constraints
model.addGenConstrPow(f_A, fA_sq, 2, "quad_fA")
model.addGenConstrPow(f_B, fB_sq, 2, "quad_fB")

# Base travel times
model.addConstr(base_TA == idle_time_A + coef_A * fA_sq, name="base_T_A")
model.addConstr(base_TB == idle_time_B + coef_B * fB_sq, name="base_T_B")

# Delayed travel times (5% increase)
model.addConstr(delay_TA == (1 + additional_rate) * base_TA, name="delay_T_A")
model.addConstr(delay_TB == (1 + additional_rate) * base_TB, name="delay_T_B")

# Indicator constraints for Road A
# y_A = 1 if f_A >= 900
model.addGenConstrIndicator(y_A, 1, f_A >= threshold_A, name="indicator_A_1")
model.addGenConstrIndicator(y_A, 0, f_A <= threshold_A - 1, name="indicator_A_0")  # strict < 900

# T_A = delay_TA if y_A=1, else T_A = base_TA
model.addGenConstrIndicator(y_A, 1, T_A == delay_TA, name="set_TAdelay")
model.addGenConstrIndicator(y_A, 0, T_A == base_TA, name="set_TAbase")

# Indicator constraints for Road B
# y_B = 1 if f_B >= 1080
model.addGenConstrIndicator(y_B, 1, f_B >= threshold_B, name="indicator_B_1")
model.addGenConstrIndicator(y_B, 0, f_B <= threshold_B - 1, name="indicator_B_0")  # strict < 1080

# T_B = delay_TB if y_B=1, else T_B = base_TB
model.addGenConstrIndicator(y_B, 1, T_B == delay_TB, name="set_TBdelay")
model.addGenConstrIndicator(y_B, 0, T_B == base_TB, name="set_TBbase")

# ========== Objective Function ==========
total_time = f_A * T_A + f_B * T_B
model.setObjective(total_time, sense=gp.GRB.MINIMIZE)

# ========== Solve and Print Results ==========
model.optimize()

if model.status == gp.GRB.OPTIMAL:
    print("\n=== Optimal Solution ===")
    print(f"f_A (vehicles on Road A) = {f_A.X:.2f}")
    print(f"f_B (vehicles on Road B) = {f_B.X:.2f}")
    print(f"T_A (travel time per vehicle on A) = {T_A.X:.4f} minutes")
    print(f"T_B (travel time per vehicle on B) = {T_B.X:.4f} minutes")
    print(f"Total travel time (objective) = {model.ObjVal:.2f} vehicle-minutes")
    print(f"Indicator y_A (f_A >= 900?) = {y_A.X}")
    print(f"Indicator y_B (f_B >= 1080?) = {y_B.X}")
    print(f"FinalAnswer=【{f_A.X:.2f}】")
else:
    print(f"Model terminated with status {model.status}")
    print(f"FinalAnswer=【No optimal solution found】")