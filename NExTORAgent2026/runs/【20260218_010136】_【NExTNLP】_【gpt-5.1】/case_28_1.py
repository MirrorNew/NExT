import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Define all parameters
# =========================

# Parameters List (must use given values)
flow_threshold_A = 900                           # integer
capacity_threshold = 0.9                         # float
year_beijing_data = 2023                         # integer (not used in model)
density_increase_step = 0.1                      # float (not used in model)
year_amap_model_upgrade = 2024                   # integer (not used in model)
prediction_capacity_threshold = 0.9              # float (not used in model)
time_reduction_rate = 0.083                      # float (not used in model)
utilization_optimal_range = [0.85, 0.89]         # list (not used in model)
total_vehicles = 2100                            # integer

idle_time = {'A': 10, 'B': 12}                   # tuple-like dict
travel_time_coefs = {                            # tuple-like dict
    'A': {'base': 10, 'coef': 0.0001},
    'B': {'base': 12, 'coef': 8e-05}
}
additional_wait_rate = 0.05                      # float
capacity = {'A': 1000, 'B': 1200}                # tuple-like dict

# Derived thresholds using given parameters
threshold_A = capacity_threshold * capacity['A']           # 0.9 * 1000 = 900
threshold_B = capacity_threshold * capacity['B']           # 0.9 * 1200 = 1080

# For indicator constraints we will use integer bounds as described
threshold_A_low = flow_threshold_A - 1          # 899
threshold_B_int = int(threshold_B)             # 1080
threshold_B_low = threshold_B_int - 1          # 1079

# =========================
# 2. Create model
# =========================

model = gp.Model("Theresia_Traffic_Allocation")

# Allow non-convexity (quadratics with indicators)
model.Params.NonConvex = 2

# =========================
# 3. Decision variables
# =========================

# Flows
f_A = model.addVar(lb=0.0, ub=capacity['A'], vtype=GRB.CONTINUOUS, name="f_A")
f_B = model.addVar(lb=0.0, ub=capacity['B'], vtype=GRB.CONTINUOUS, name="f_B")

# Travel times per vehicle
T_A = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="T_A")
T_B = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="T_B")

# Base travel time expressions
t_A_base = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="t_A_base")
t_B_base = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="t_B_base")

# Binary indicators for 90% capacity extra delay
y_A = model.addVar(vtype=GRB.BINARY, name="y_A")
y_B = model.addVar(vtype=GRB.BINARY, name="y_B")

# =========================
# 4. Auxiliary substitution variables
#    to model squares f_A^2 and f_B^2
# =========================

fA_sq = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="fA_sq")
fB_sq = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="fB_sq")

# (No need for explicit -INF/+INF aux vars here; quadratics are handled via gen-constr)

# =========================
# 5. Base quadratic relationships
# =========================

# fA_sq = f_A^2
model.addGenConstrPow(f_A, fA_sq, 2.0, name="pow_fA_sq")

# fB_sq = f_B^2
model.addGenConstrPow(f_B, fB_sq, 2.0, name="pow_fB_sq")

# Link base times linearly using travel_time_coefs
model.addConstr(
    t_A_base == travel_time_coefs['A']['base'] +
    travel_time_coefs['A']['coef'] * fA_sq,
    name="base_time_A"
)

model.addConstr(
    t_B_base == travel_time_coefs['B']['base'] +
    travel_time_coefs['B']['coef'] * fB_sq,
    name="base_time_B"
)

# =========================
# 6. Add constraints
# =========================

# 6.1 Flow conservation
model.addConstr(f_A + f_B == total_vehicles, name="flow_conservation")

# 6.2 Capacity bounds are already in variable definitions (lb, ub),
#     but we can add explicit constraints for clarity.
model.addConstr(f_A <= capacity['A'], name="capacity_A")
model.addConstr(f_B <= capacity['B'], name="capacity_B")

# 6.3 Indicator constraints for 90% capacity thresholds

# Road A: y_A = 1  -> f_A >= 900
model.addGenConstrIndicator(y_A, 1, f_A >= threshold_A, name="A_above_90pct")

# Road A: y_A = 0  -> f_A <= 899
model.addGenConstrIndicator(y_A, 0, f_A <= threshold_A_low, name="A_below_90pct")

# Road B: y_B = 1  -> f_B >= 1080
model.addGenConstrIndicator(y_B, 1, f_B >= threshold_B_int, name="B_above_90pct")

# Road B: y_B = 0  -> f_B <= 1079
model.addGenConstrIndicator(y_B, 0, f_B <= threshold_B_low, name="B_below_90pct")

# 6.4 Indicator constraints for travel time with extra delay

# Extra rate factor = 1 + additional_wait_rate = 1.05
extra_factor = 1.0 + additional_wait_rate

# Road A:
# y_A = 0 -> T_A = t_A_base
model.addGenConstrIndicator(y_A, 0, T_A == t_A_base, name="A_time_no_extra")

# y_A = 1 -> T_A = 1.05 * t_A_base
model.addGenConstrIndicator(y_A, 1, T_A == extra_factor * t_A_base, name="A_time_with_extra")

# Road B:
# y_B = 0 -> T_B = t_B_base
model.addGenConstrIndicator(y_B, 0, T_B == t_B_base, name="B_time_no_extra")

# y_B = 1 -> T_B = 1.05 * t_B_base
model.addGenConstrIndicator(y_B, 1, T_B == extra_factor * t_B_base, name="B_time_with_extra")

# =========================
# 7. Objective function
# =========================

# Minimize total travel time of all vehicles: Z = f_A * T_A + f_B * T_B
model.setObjective(f_A * T_A + f_B * T_B, GRB.MINIMIZE)

# =========================
# 8. Solve model
# =========================

model.optimize()

# =========================
# 9. Print results
# =========================

if model.status == GRB.OPTIMAL:
    f_A_opt = f_A.X
    f_B_opt = f_B.X
    T_A_opt = T_A.X
    T_B_opt = T_B.X
    Z_opt = model.ObjVal

    print("Optimal solution found:")
    print(f"  f_A (vehicles on Road A) = {f_A_opt:.4f}")
    print(f"  f_B (vehicles on Road B) = {f_B_opt:.4f}")
    print(f"  T_A (min)                = {T_A_opt:.4f}")
    print(f"  T_B (min)                = {T_B_opt:.4f}")
    print(f"  Total travel time Z      = {Z_opt:.4f}")
    print(f"  y_A (A >= 90% cap?)      = {int(y_A.X)}")
    print(f"  y_B (B >= 90% cap?)      = {int(y_B.X)}")
else:
    print(f"Optimization ended with status {model.status}")
    f_A_opt = None

# =========================
# 10. FinalAnswer output
# =========================
# The question asks: determine the number of vehicles f_A assigned to road A.
# So FinalAnswer is f_A_opt.

print(f"FinalAnswer=【{f_A_opt}】")