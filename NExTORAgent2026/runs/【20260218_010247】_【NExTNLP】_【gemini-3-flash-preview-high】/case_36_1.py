import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter values from the list
h_ideal = 100.0
h_threshold = 90.0
h_min_downstream = 10.0
V_storage_raw = 100000000.0
V_storage = V_storage_raw / 1000000.0  # Convert to million cubic meters
h_initial = 100.0
drop_rate = 1.25
gen_coefficient = 5.0
consumption_coeff = 0.5

# 2. Create the Gurobi model
model = gp.Model("Hydroelectric_Optimization")
model.Params.NonConvex = 2  # Required for bilinear constraints and non-convex terms

# 3. Create decision variables as defined in Context
V = model.addVar(lb=0, ub=100, name="V")  # Water release volume (million cubic meters)
h = model.addVar(lb=10, ub=100, name="h")  # Reservoir head height (meters)
Pu = model.addVar(lb=275, ub=500, name="Pu")  # Unit water power generation
P = model.addVar(lb=0, ub=50000, name="P")  # Total power generation
delta_plus = model.addVar(lb=0, ub=80, name="delta_plus")  # Head deficiency auxiliary variable
C = model.addVar(lb=0, ub=3200, name="C")  # Additional energy consumption penalty

# 4. Create auxiliary substitution and indicator variables from coding advice
# These variables range from -infinity to infinity per instructions
sq_delta_plus = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="sq_delta_plus")
y = model.addVar(vtype=GRB.BINARY, name="y")  # Indicator binary variable for head deficiency threshold

# 5. Set up the objective function
# Maximize power generation revenue (Total generation - additional consumption penalty)
model.setObjective(P - C, GRB.MAXIMIZE)

# 6. Add all constraints
# Volume capacity and Head-volume relation
model.addConstr(h == h_initial - drop_rate * V, name="head_volume_relation")

# Unit power generation definition: P_u = 5 * (initial_head + current_head) / 2
model.addConstr(Pu == gen_coefficient * (h_initial + h) / 2, name="unit_generation_def")

# Total power generation: P = P_u * V (Bilinear term handled by NonConvex=2)
model.addConstr(P == Pu * V, name="total_generation_bilinear")

# Additional consumption logic: C = 0.5 * (delta_plus)^2
model.addGenConstrPow(delta_plus, sq_delta_plus, 2)
model.addConstr(C == consumption_coeff * sq_delta_plus, name="consumption_penalty_def")

# Head threshold logic using indicators
# Case 1: y = 1 implies head height h <= 90
model.addGenConstrIndicator(y, 1, h <= h_threshold)
# Case 2: y = 0 implies head height h >= 90
model.addGenConstrIndicator(y, 0, h >= h_threshold)

# When head is below threshold (y=1), delta_plus = 90 - h
model.addGenConstrIndicator(y, 1, delta_plus == h_threshold - h)
# When head is above threshold (y=0), delta_plus = 0
model.addGenConstrIndicator(y, 0, delta_plus == 0)

# 7. Solve the model
model.optimize()

# Print results
if model.Status == GRB.OPTIMAL:
    objective_value = model.ObjVal
    print(f"Optimal Water Release Volume V: {V.X}")
    print(f"Optimal Resulting Head h: {h.X}")
    print(f"Maximum Revenue Z: {objective_value}")
    print(f"FinalAnswer=【{objective_value}】")
else:
    print("Optimization was not successful.")