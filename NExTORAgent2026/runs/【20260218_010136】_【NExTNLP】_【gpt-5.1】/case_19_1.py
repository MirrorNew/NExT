import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters and Data
# =========================
total_water_resources = 100
number_of_farms = 3
a = [5, 3, 4]  # a_j coefficients, j = 1..3
total_irrigation_amount_limit = 100
sum_of_squares_of_irrigation_water_limit = 3500

# =========================
# 2. Create Model
# =========================
model = gp.Model("Irrigation_Optimization")

# Allow nonconvex general constraints (powers with fractional exponents)
model.Params.NonConvex = 2

# =========================
# 3. Decision Variables
# =========================
# w[j]: irrigation water allocated to farm j (j = 0..2 corresponds to farms 1..3)
w = model.addVars(number_of_farms, lb=0.0, vtype=GRB.CONTINUOUS, name="w")

# =========================
# 4. Auxiliary Variables
# =========================
# y[j]: nonlinear yield terms corresponding to w_j^(j/4)
# z[j]: squared terms corresponding to w_j^2
# NOTE: auxiliary variables have full range (-inf, +inf) as required
y = model.addVars(number_of_farms, lb=-GRB.INFINITY, ub=GRB.INFINITY,
                  vtype=GRB.CONTINUOUS, name="y")
z = model.addVars(number_of_farms, lb=-GRB.INFINITY, ub=GRB.INFINITY,
                  vtype=GRB.CONTINUOUS, name="z")

# =========================
# 5. Objective Function
# =========================
# Maximize total yield: sum_j a_j * y_j, where y_j = w_j^(j/4)
obj_expr = gp.LinExpr()
for j in range(number_of_farms):
    obj_expr += a[j] * y[j]

model.setObjective(obj_expr, GRB.MAXIMIZE)

# =========================
# 6. Constraints
# =========================

# 6.1 Power constraints for yield components:
# j index in math is 1..3, but here it's 0..2, so exponent is (j+1)/4.
for j in range(number_of_farms):
    exponent_y = (j + 1) / 4.0  # 1/4, 2/4, 3/4
    model.addGenConstrPow(w[j], y[j], exponent_y, name=f"yield_pow_{j+1}")

# 6.2 Power constraints for quadratic water-usage terms: z_j = w_j^2
for j in range(number_of_farms):
    model.addGenConstrPow(w[j], z[j], 2.0, name=f"quad_pow_{j+1}")

# 6.3 Total water availability: sum_j w_j <= total_irrigation_amount_limit
model.addConstr(
    gp.quicksum(w[j] for j in range(number_of_farms)) <= total_irrigation_amount_limit,
    name="total_water_limit"
)

# 6.4 Quadratic water-usage limit: sum_j z_j <= sum_of_squares_of_irrigation_water_limit
model.addConstr(
    gp.quicksum(z[j] for j in range(number_of_farms)) <= sum_of_squares_of_irrigation_water_limit,
    name="sum_of_squares_limit"
)

# 6.5 Nonnegativity constraints (explicit, though lb=0 already enforces this)
for j in range(number_of_farms):
    model.addConstr(w[j] >= 0.0, name=f"nonneg_w_{j+1}")

# =========================
# 7. Solve the Model
# =========================
model.optimize()

# =========================
# 8. Print Results
# =========================
if model.Status == GRB.OPTIMAL:
    print("Optimal solution found.")
    for j in range(number_of_farms):
        print(f"w_{j+1} (water to farm {j+1}) = {w[j].X}")
    print(f"Objective value (total yield) Z = {model.ObjVal}")
    final_answer_value = model.ObjVal
else:
    print(f"Optimization ended with status {model.Status}")
    final_answer_value = float('nan')

# =========================
# Final Answer Output
# =========================
print(f"FinalAnswer=【{final_answer_value}】")