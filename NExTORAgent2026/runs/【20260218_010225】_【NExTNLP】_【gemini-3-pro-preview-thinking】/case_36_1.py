import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and any other necessary packages.
# (Already imported above)

# 2. Define all parameter matrices and data inputs.
# Parameters from the provided Parameters List
h_ideal = 100.0
h_threshold = 90.0
h_min_downstream = 10.0
V_storage = 100000000.0  # in cubic meters
h_initial = 100.0
drop_rate = 1.25
gen_coefficient = 5.0
consumption_coeff = 0.5

# Convert Volume capacity to millions of cubic meters to match drop_rate units
V_max_millions = V_storage / 1000000.0 

# 3. Create decision variables.
model = gp.Model("Hydropower_Optimization")
model.Params.NonConvex = 2  # Enable handling of non-convex quadratic constraints

# V: Water release volume (in million cubic meters)
# Range 0 <= V <= 100 (derived from V_max_millions)
V = model.addVar(lb=0.0, ub=V_max_millions, vtype=GRB.CONTINUOUS, name="V")

# h: Reservoir head height
# Range 10 <= h <= 100
h = model.addVar(lb=h_min_downstream, ub=h_initial, vtype=GRB.CONTINUOUS, name="h")

# P_u: Unit water power generation
P_u = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="P_u")

# P: Total power generation
P = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="P")

# C: Additional energy consumption penalty
C = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="C")

# 4. Create any auxiliary substitution or indicator variables.
# Delta_plus: represents (90 - h)^+
Delta_plus = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="Delta_plus")

# sq_Delta: represents (Delta_plus)^2
sq_Delta = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="sq_Delta")

# 5. Set up the objective function.
# Maximize Z = P - C
model.setObjective(P - C, GRB.MAXIMIZE)

# 6. Add all constraints.

# Constraint: Head-Volume relation
# h = 100 - 1.25 * V
model.addConstr(h == h_initial - drop_rate * V, name="Head_Volume_Relation")

# Constraint: Unit power generation definition
# P_u = 5 * (100 + h) / 2
model.addConstr(P_u == gen_coefficient * (h_initial + h) / 2, name="Unit_Power_Gen")

# Constraint: Total power generation
# P = P_u * V
model.addConstr(P == P_u * V, name="Total_Power_Calc")

# Constraint: Penalty auxiliary definition
# Delta_plus >= 90 - h  (and Delta_plus >= 0 from variable bounds)
# This combined with minimizing C (in the objective) ensures Delta_plus = max(0, 90-h)
model.addConstr(Delta_plus >= h_threshold - h, name="Delta_Plus_Lower_Bound")

# Constraint: Additional consumption definition
# C = 0.5 * (Delta_plus)^2
# We use General Constraint for Power: sq_Delta = Delta_plus ^ 2
model.addGenConstrPow(Delta_plus, sq_Delta, 2, name="Square_Delta_GenConstr")
model.addConstr(C == consumption_coeff * sq_Delta, name="Consumption_Penalty_Calc")

# 7. Solve the model and print results.
model.optimize()

if model.Status == GRB.OPTIMAL:
    print(f"Optimal objective value: {model.ObjVal}")
    print(f"Optimal Water Release V: {V.X}")
    print(f"Resulting Head h: {h.X}")
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimization was not successful.")
    print("FinalAnswer=【No Solution】")