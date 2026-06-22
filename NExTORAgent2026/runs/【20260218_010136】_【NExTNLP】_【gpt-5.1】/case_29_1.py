import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Define parameters ONLY from Parameters List
# =========================
threshold_speed_extra = 50                     # integer
number_of_sections = 3                         # integer
base_fuel_consumption_coefficient = 0.005      # float
extra_fuel_consumption_coefficient = 0.0001    # float
total_sailing_time_limit = 10                  # integer
min_speed_section_2 = 55                       # integer

Table_1_SectionCharacteristics = [
    {'Section': 1, 'Distance_km': 200, 'Fuel_price_yuan_per_ton': 3000, 'Max_speed_kmh': 70},
    {'Section': 2, 'Distance_km': 100, 'Fuel_price_yuan_per_ton': 3500, 'Max_speed_kmh': 60},
    {'Section': 3, 'Distance_km': 150, 'Fuel_price_yuan_per_ton': 4000, 'Max_speed_kmh': 80}
]

# Extract distances, fuel prices, and max speeds from the table
D = [sec['Distance_km'] for sec in Table_1_SectionCharacteristics]
p = [sec['Fuel_price_yuan_per_ton'] for sec in Table_1_SectionCharacteristics]
vmax = [sec['Max_speed_kmh'] for sec in Table_1_SectionCharacteristics]

# =========================
# 2. Create model
# =========================
model = gp.Model("Malacca_Speed_Optimization")

# Allow non-convex quadratic and bilinear constraints
model.Params.NonConvex = 2

# =========================
# 3. Decision variables
# =========================
# v1, v2, v3: speeds in each section
v1 = model.addVar(lb=0.0, ub=vmax[0], vtype=GRB.CONTINUOUS, name="v1")
v2 = model.addVar(lb=min_speed_section_2, ub=vmax[1], vtype=GRB.CONTINUOUS, name="v2")
v3 = model.addVar(lb=0.0, ub=vmax[2], vtype=GRB.CONTINUOUS, name="v3")

# d1, d2, d3: extra speed above threshold_speed_extra
d1 = model.addVar(lb=0.0, ub=vmax[0] - threshold_speed_extra, vtype=GRB.CONTINUOUS, name="d1")
d2 = model.addVar(lb=0.0, ub=vmax[1] - threshold_speed_extra, vtype=GRB.CONTINUOUS, name="d2")
d3 = model.addVar(lb=0.0, ub=vmax[2] - threshold_speed_extra, vtype=GRB.CONTINUOUS, name="d3")

# =========================
# 4. Auxiliary variables
# =========================
# Squares of speeds: s1 = v1^2, s2 = v2^2, s3 = v3^2
s1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="s1")
s2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="s2")
s3 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="s3")

# Squares of extra speeds: e1 = d1^2, e2 = d2^2, e3 = d3^2
e1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="e1")
e2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="e2")
e3 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="e3")

# Time variables for each section: t1 = D1/v1, etc.
t1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="t1")
t2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="t2")
t3 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="t3")

# Reciprocal helper variables: z1, z2, z3 such that v1*z1=1, etc.
z1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z1")
z2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z2")
z3 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z3")

# =========================
# 5. General constraints for powers and reciprocals
# =========================
# Squares of speeds
model.addGenConstrPow(v1, s1, 2.0, name="s1_def")
model.addGenConstrPow(v2, s2, 2.0, name="s2_def")
model.addGenConstrPow(v3, s3, 2.0, name="s3_def")

# Squares of extra speeds
model.addGenConstrPow(d1, e1, 2.0, name="e1_def")
model.addGenConstrPow(d2, e2, 2.0, name="e2_def")
model.addGenConstrPow(d3, e3, 2.0, name="e3_def")

# Reciprocal relations: vi * zi = 1 and ti = Di * zi
model.addConstr(v1 * z1 == 1, name="recip_v1")
model.addConstr(t1 == D[0] * z1, name="time1_def")

model.addConstr(v2 * z2 == 1, name="recip_v2")
model.addConstr(t2 == D[1] * z2, name="time2_def")

model.addConstr(v3 * z3 == 1, name="recip_v3")
model.addConstr(t3 == D[2] * z3, name="time3_def")

# =========================
# 6. Constraints
# =========================

# Extra-speed definition: di >= vi - threshold_speed_extra
model.addConstr(d1 >= v1 - threshold_speed_extra, name="d1_def")
model.addConstr(d2 >= v2 - threshold_speed_extra, name="d2_def")
model.addConstr(d3 >= v3 - threshold_speed_extra, name="d3_def")

# (Bounds on d1,d2,d3 already handled by variable definitions)

# Time constraint: total time <= total_sailing_time_limit
model.addConstr(t1 + t2 + t3 <= total_sailing_time_limit, name="total_time")

# (Speed limits and min speed for section 2 already in variable bounds)

# =========================
# 7. Objective function
# =========================
# Z = sum_i p_i * D_i * (alpha * v_i^2 + beta * d_i^2)
alpha = base_fuel_consumption_coefficient
beta = extra_fuel_consumption_coefficient

obj_expr = (
    p[0] * D[0] * (alpha * s1 + beta * e1) +
    p[1] * D[1] * (alpha * s2 + beta * e2) +
    p[2] * D[2] * (alpha * s3 + beta * e3)
)

model.setObjective(obj_expr, GRB.MINIMIZE)

# =========================
# 8. Optimize
# =========================
model.optimize()

# =========================
# 9. Print results
# =========================
if model.Status == GRB.OPTIMAL:
    v1_val = v1.X
    v2_val = v2.X
    v3_val = v3.X
    d1_val = d1.X
    d2_val = d2.X
    d3_val = d3.X
    total_time = t1.X + t2.X + t3.X
    total_cost = model.ObjVal

    print("Optimal solution found:")
    print(f"v1 (section 1 speed) = {v1_val:.6f} km/h")
    print(f"v2 (section 2 speed) = {v2_val:.6f} km/h")
    print(f"v3 (section 3 speed) = {v3_val:.6f} km/h")
    print(f"d1 (extra speed section 1) = {d1_val:.6f} km/h")
    print(f"d2 (extra speed section 2) = {d2_val:.6f} km/h")
    print(f"d3 (extra speed section 3) = {d3_val:.6f} km/h")
    print(f"Total sailing time = {total_time:.6f} hours")
    print(f"Total fuel cost (objective) = {total_cost:.6f} yuan")

    # FinalAnswer is defined as the minimum total fuel cost
    print(f"FinalAnswer=【{total_cost:.6f}】")
else:
    print(f"Optimization ended with status {model.Status}")
    # If no optimal solution, we still output something for FinalAnswer
    print("FinalAnswer=【NaN】")