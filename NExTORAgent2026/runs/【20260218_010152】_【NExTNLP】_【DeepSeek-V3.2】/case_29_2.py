import gurobipy as gp

# Define parameters
threshold_speed_extra = 50
number_of_sections = 3
base_coef = 0.005
extra_coef = 0.0001
total_time_limit = 10
min_speed_section_2 = 55

section_data = [
    {'Section': 1, 'Distance_km': 200, 'Fuel_price_yuan_per_ton': 3000, 'Max_speed_kmh': 70},
    {'Section': 2, 'Distance_km': 100, 'Fuel_price_yuan_per_ton': 3500, 'Max_speed_kmh': 60},
    {'Section': 3, 'Distance_km': 150, 'Fuel_price_yuan_per_ton': 4000, 'Max_speed_kmh': 80}
]

# Create model
model = gp.Model("Malacca_Speed_Optimization")
model.Params.NonConvex = 2

# Decision variables
v = []
d = []
y = []
for i in range(number_of_sections):
    if i == 1:  # Section 2 has minimum speed constraint
        v_i = model.addVar(lb=min_speed_section_2, ub=section_data[i]['Max_speed_kmh'], 
                          name=f"v_{i+1}")
    else:  # Sections 1 and 3 have lower bound 0
        v_i = model.addVar(lb=0, ub=section_data[i]['Max_speed_kmh'], 
                          name=f"v_{i+1}")
    v.append(v_i)
    
    d_max = section_data[i]['Max_speed_kmh'] - threshold_speed_extra
    d_i = model.addVar(lb=0, ub=d_max, name=f"d_{i+1}")
    d.append(d_i)
    
    y_i = model.addVar(vtype=gp.GRB.BINARY, name=f"y_{i+1}")
    y.append(y_i)

# Auxiliary variables for time denominators
t = []
for i in range(number_of_sections):
    t_i = model.addVar(lb=0, ub=gp.GRB.INFINITY, name=f"t_{i+1}")
    t.append(t_i)

# Time constraints: v_i * t_i = distance_i
model.addConstr(v[0] * t[0] == 200, "time_constraint_section1")
model.addConstr(v[1] * t[1] == 100, "time_constraint_section2")
model.addConstr(v[2] * t[2] == 150, "time_constraint_section3")

# Total time constraint
model.addConstr(t[0] + t[1] + t[2] <= total_time_limit, "total_time_limit")

# Indicator constraints for extra speed
epsilon = 1e-3  # small tolerance
for i in range(number_of_sections):
    model.addGenConstrIndicator(y[i], 1, v[i] >= threshold_speed_extra + epsilon,
                               f"indicator_{i+1}_v_ge_50")
    model.addGenConstrIndicator(y[i], 0, v[i] <= threshold_speed_extra,
                               f"indicator_{i+1}_v_le_50")
    
    # d_i >= v_i - threshold_speed_extra
    model.addConstr(d[i] >= v[i] - threshold_speed_extra, f"d_def_lower_{i+1}")
    
    # d_i <= (max_speed_i - threshold_speed_extra) * y_i
    d_max_i = section_data[i]['Max_speed_kmh'] - threshold_speed_extra
    model.addConstr(d[i] <= d_max_i * y[i], f"d_def_upper_{i+1}")

# Auxiliary variables for squares
v_sq = []
for i in range(number_of_sections):
    v_sq_i = model.addVar(lb=0, ub=gp.GRB.INFINITY, name=f"v_sq_{i+1}")
    v_sq.append(v_sq_i)
    model.addGenConstrPow(v[i], v_sq_i, 2, f"pow_v_{i+1}")

d_sq = []
for i in range(number_of_sections):
    d_sq_i = model.addVar(lb=0, ub=gp.GRB.INFINITY, name=f"d_sq_{i+1}")
    d_sq.append(d_sq_i)
    model.addGenConstrPow(d[i], d_sq_i, 2, f"pow_d_{i+1}")

# Objective function
obj = gp.QuadExpr()
for i in range(number_of_sections):
    distance = section_data[i]['Distance_km']
    price = section_data[i]['Fuel_price_yuan_per_ton']
    obj += price * distance * (base_coef * v_sq[i] + extra_coef * d_sq[i])

model.setObjective(obj, gp.GRB.MINIMIZE)

# Solve
model.optimize()

# Print results
print("Optimal Solution:")
print(f"v1 = {v[0].X:.4f} km/h")
print(f"v2 = {v[1].X:.4f} km/h")
print(f"v3 = {v[2].X:.4f} km/h")
print(f"d1 = {d[0].X:.4f} km/h")
print(f"d2 = {d[1].X:.4f} km/h")
print(f"d3 = {d[2].X:.4f} km/h")
print(f"Total fuel cost = {model.ObjVal:.2f} yuan")

# Calculate total time
total_time = t[0].X + t[1].X + t[2].X
print(f"Total sailing time = {total_time:.4f} hours")

# Check if any section exceeds 50 km/h
for i in range(number_of_sections):
    if v[i].X > threshold_speed_extra:
        print(f"Section {i+1} exceeds 50 km/h by {v[i].X - threshold_speed_extra:.2f} km/h")

print(f"FinalAnswer=【{model.ObjVal}】")