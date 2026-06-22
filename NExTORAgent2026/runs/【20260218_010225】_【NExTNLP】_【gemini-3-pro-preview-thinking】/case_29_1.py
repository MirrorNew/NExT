import gurobipy as gp
from gurobipy import GRB

# Define the parameters based on the provided Parameters List
threshold_speed_extra = 50
number_of_sections = 3
base_fuel_consumption_coefficient = 0.005
extra_fuel_consumption_coefficient = 0.0001
total_sailing_time_limit = 10
min_speed_section_2 = 55

# Section characteristics provided in the list
# We can organize them for easy iteration
# Section 1: Dist 200, Price 3000, Max 70
# Section 2: Dist 100, Price 3500, Max 60 (Min 55 from separate param)
# Section 3: Dist 150, Price 4000, Max 80
section_data = [
    {'id': 1, 'dist': 200, 'price': 3000, 'max_v': 70, 'min_v': 0},
    {'id': 2, 'dist': 100, 'price': 3500, 'max_v': 60, 'min_v': min_speed_section_2},
    {'id': 3, 'dist': 150, 'price': 4000, 'max_v': 80, 'min_v': 0}
]

# Create the Gurobi model
model = gp.Model("ShipSpeedOptimization")

# Set NonConvex parameter to 2 to handle quadratic equality constraints (v * t = dist)
model.Params.NonConvex = 2

# --- Decision Variables ---
v = {}      # Speed for each section
d = {}      # Extra speed above 50 km/h
t = {}      # Time taken for each section (auxiliary substitution for 1/v)
v_sq = {}   # Square of speed (auxiliary)
d_sq = {}   # Square of extra speed (auxiliary)

for sec in section_data:
    i = sec['id']
    
    # Speed variable v_i
    v[i] = model.addVar(lb=sec['min_v'], ub=sec['max_v'], vtype=GRB.CONTINUOUS, name=f"v_{i}")
    
    # Extra speed variable d_i
    # Range is technically [0, max_v - 50], but 0 to infinity covers it given the constraints.
    d[i] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"d_{i}")
    
    # Time variable t_i
    t[i] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"t_{i}")
    
    # Squared variables for objective function
    v_sq[i] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"v_sq_{i}")
    d_sq[i] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"d_sq_{i}")

# --- Constraints ---

total_time_expr = 0

for sec in section_data:
    i = sec['id']
    dist = sec['dist']
    
    # 1. Time definition: v_i * t_i == Distance_i
    # This eliminates the 1/v term in the denominator.
    model.addConstr(v[i] * t[i] == dist, name=f"Time_Def_Section_{i}")
    
    # 2. Extra speed definition: d_i >= v_i - 50
    # Since we minimize cost (positive coefficient for d^2), d_i will be 0 if v_i <= 50.
    model.addConstr(d[i] >= v[i] - threshold_speed_extra, name=f"Extra_Speed_Constraint_{i}")
    
    # 3. Auxiliary quadratic equalities
    # v_sq = v^2
    model.addGenConstrPow(v[i], v_sq[i], 2, name=f"GenConstr_v_sq_{i}")
    # d_sq = d^2
    model.addGenConstrPow(d[i], d_sq[i], 2, name=f"GenConstr_d_sq_{i}")
    
    # Sum up time for total time constraint
    total_time_expr += t[i]

# 4. Total Sailing Time Constraint
model.addConstr(total_time_expr <= total_sailing_time_limit, name="Total_Sailing_Time_Limit")

# --- Objective Function ---
# Minimize Z = sum [ P_i * D_i * (0.005 * v_i^2 + 0.0001 * d_i^2) ]
objective_expr = 0

for sec in section_data:
    i = sec['id']
    price = sec['price']
    dist = sec['dist']
    
    # Cost for this section
    # Base consumption component: 0.005 * v^2
    # Extra consumption component: 0.0001 * d^2
    # Total Cost = Price * Distance * Consumption_Rate
    
    section_cost = price * dist * (base_fuel_consumption_coefficient * v_sq[i] + extra_fuel_consumption_coefficient * d_sq[i])
    objective_expr += section_cost

model.setObjective(objective_expr, GRB.MINIMIZE)

# Solve the model
model.optimize()

# Output the result
if model.Status == GRB.OPTIMAL:
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimization was unsuccessful.")