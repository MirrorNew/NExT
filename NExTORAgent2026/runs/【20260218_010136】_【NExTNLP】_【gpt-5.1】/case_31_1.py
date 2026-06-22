import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Define all parameters
# =========================

# Parameters from the provided Parameters List
threshold_salinity_field2   = 80        # not directly used in this model, but kept for completeness
total_fertilizer_capacity   = 150
max_fertilizer_field1       = 100
max_fertilizer_field2       = 90
yield_coeff_field1          = 5
yield_coeff_field2          = 6
reduction_coef_field2       = 2e-05

# =========================
# 2. Create model
# =========================

model = gp.Model("Kazdale_Farm_Fertilizer_Allocation")

# Allow nonconvex constraints (due to pow with exponent 0.5 and 2.0)
model.Params.NonConvex = 2

# =========================
# 3. Create decision variables
# =========================

# Fertilizer allocations (kg)
x1 = model.addVar(lb=0.0, ub=max_fertilizer_field1, vtype=GRB.CONTINUOUS, name="x1")
x2 = model.addVar(lb=0.0, ub=max_fertilizer_field2, vtype=GRB.CONTINUOUS, name="x2")

# Yields (tons)
y1 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="y1")
y2 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="y2")

# Reduction in Field 2 yield (tons)
Delta2 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="Delta2")

# =========================
# 4. Auxiliary substitution variables
# =========================
# These are free (−∞ to +∞) auxiliary variables as requested

# Auxiliary for sqrt(x1)
x1_sqrt = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                       vtype=GRB.CONTINUOUS, name="x1_sqrt")

# Auxiliary for sqrt(x2)
x2_sqrt = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                       vtype=GRB.CONTINUOUS, name="x2_sqrt")

# Auxiliary for x2^2
x2_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                     vtype=GRB.CONTINUOUS, name="x2_sq")

# =========================
# 5. Set up the objective
# =========================
# Maximize total net yield: y1 + (y2 - Delta2)

model.setObjective(y1 + y2 - Delta2, GRB.MAXIMIZE)

# =========================
# 6. Add all constraints
# =========================

# 6.1 Total fertilizer allocation
model.addConstr(x1 + x2 <= total_fertilizer_capacity, name="Total_Fertilizer")

# 6.2 Individual fertilizer upper bounds (already in variable bounds, but also as constraints)
model.addConstr(x1 <= max_fertilizer_field1, name="Max_Fertilizer_Field1")
model.addConstr(x2 <= max_fertilizer_field2, name="Max_Fertilizer_Field2")

# 6.3 Yield function Field 1: y1 = yield_coeff_field1 * sqrt(x1)
#     Implemented via auxiliary variable x1_sqrt with a pow general constraint
model.addGenConstrPow(x1, x1_sqrt, 0.5, name="Pow_x1_sqrt")
model.addConstr(y1 == yield_coeff_field1 * x1_sqrt, name="Yield_Field1")

# 6.4 Yield function Field 2: y2 = yield_coeff_field2 * sqrt(x2)
model.addGenConstrPow(x2, x2_sqrt, 0.5, name="Pow_x2_sqrt")
model.addConstr(y2 == yield_coeff_field2 * x2_sqrt, name="Yield_Field2")

# 6.5 Yield reduction Field 2: Delta2 = reduction_coef_field2 * x2^2
model.addGenConstrPow(x2, x2_sq, 2.0, name="Pow_x2_sq")
model.addConstr(Delta2 == reduction_coef_field2 * x2_sq, name="Yield_Reduction_Field2")

# =========================
# 7. Solve the model
# =========================

model.optimize()

# =========================
# 8. Print results
# =========================

if model.status == GRB.OPTIMAL:
    x1_opt = x1.X
    x2_opt = x2.X
    y1_opt = y1.X
    y2_opt = y2.X
    Delta2_opt = Delta2.X
    Z_opt = model.ObjVal

    print("Optimal solution found:")
    print(f"  x1 (fertilizer Field 1, kg)      = {x1_opt:.6f}")
    print(f"  x2 (fertilizer Field 2, kg)      = {x2_opt:.6f}")
    print(f"  y1 (yield Field 1, tons)         = {y1_opt:.6f}")
    print(f"  y2 (gross yield Field 2, tons)   = {y2_opt:.6f}")
    print(f"  Delta2 (yield reduction F2, ton) = {Delta2_opt:.6f}")
    print(f"  Total net yield (objective)      = {Z_opt:.6f}")
else:
    print(f"Optimization ended with status {model.status}")
    Z_opt = float('nan')

# =========================
# 9. Final answer output
# =========================
# The question asks to maximize total yield y1 + y2 (net for field 2 after reduction),
# which is exactly the model objective value Z_opt.

print(f"FinalAnswer=【{Z_opt}】")