import gurobipy as gp

# Parameters from the provided list
h_initial = 100.0
h_threshold = 90.0
h_min_downstream = 10.0
drop_rate = 1.25
gen_coefficient = 5.0
consumption_coeff = 0.5
V_storage = 100000000.0  # Note: V_storage is not directly used in the constraints as per the given math model

# Create model
model = gp.Model("HydropowerOptimization")
model.Params.NonConvex = 2

# Decision Variables
V = model.addVar(lb=0.0, ub=100.0, name="V")
h = model.addVar(lb=10.0, ub=100.0, name="h")
Pu = model.addVar(lb=275.0, ub=500.0, name="Pu")
P = model.addVar(lb=0.0, ub=50000.0, name="P")
Delta_plus = model.addVar(lb=0.0, ub=80.0, name="Delta_plus")
C = model.addVar(lb=0.0, ub=3200.0, name="C")

# Auxiliary Variables
y = model.addVar(vtype=gp.GRB.BINARY, name="y")  # indicator for h < 90
W = model.addVar(lb=0.0, ub=6400.0, name="W")  # (Delta_plus)^2
X = model.addVar(lb=110.0, ub=200.0, name="X")  # 100 + h
Y = model.addVar(lb=55.0, ub=100.0, name="Y")  # X/2

# Head-volume relation
model.addConstr(h == h_initial - drop_rate * V, "head_volume")

# Unit power generation: Pu = 5 * (100 + h)/2
model.addConstr(X == 100 + h, "compute_X")
model.addConstr(Y == X / 2, "compute_Y")
model.addConstr(Pu == gen_coefficient * Y, "unit_power")

# Total power generation
model.addConstr(P == Pu * V, "total_power")

# Indicator constraints for Delta_plus = (90 - h) if h < 90, else 0
epsilon = 1e-6  # small tolerance for strict inequality
# y = 1 if h <= 89.999 (i.e., h < 90)
model.addGenConstrIndicator(y, 1, h <= h_threshold - epsilon)
# y = 0 if h >= 90
model.addGenConstrIndicator(y, 0, h >= h_threshold)

# Delta_plus = (90 - h) * y
model.addConstr(Delta_plus == (h_threshold - h) * y, "Delta_plus_def")

# Quadratic penalty: C = 0.5 * (Delta_plus)^2
model.addGenConstrPow(Delta_plus, W, 2, "square_Delta")
model.addConstr(C == consumption_coeff * W, "penalty_cost")

# Objective: maximize Z = P - C
model.setObjective(P - C, gp.GRB.MAXIMIZE)

# Solve
model.optimize()

# Output results
if model.status == gp.GRB.OPTIMAL:
    V_val = V.x
    h_val = h.x
    P_val = P.x
    print(f"Optimal water release volume V = {V_val:.2f} million m³")
    print(f"Resulting head height h = {h_val:.2f} m")
    print(f"Maximum total power generation P = {P_val:.2f}")
    print(f"FinalAnswer=【{P_val:.2f}】")
else:
    print("No optimal solution found.")
    print(f"FinalAnswer=【None】")