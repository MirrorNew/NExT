import gurobipy as gp
from gurobipy import GRB

# ----------------------------
# 2. Define parameters
# ----------------------------
total_fertilizer = 150          # total_fertilizer_capacity
max_f1 = 100                    # max_fertilizer_field1
max_f2 = 90                     # max_fertilizer_field2
coeff_f1 = 5                    # yield_coeff_field1
coeff_f2 = 6                    # yield_coeff_field2
reduction_coeff = 2e-05         # reduction_coef_field2

# ----------------------------
# 3. Create model
# ----------------------------
model = gp.Model("FertilizerAllocation")
model.Params.NonConvex = 2      # for sqrt and quadratic constraints

# ----------------------------
# 4. Decision variables
# ----------------------------
x1 = model.addVar(lb=0, ub=max_f1, name="x1")      # fertilizer field 1
x2 = model.addVar(lb=0, ub=max_f2, name="x2")      # fertilizer field 2
y1 = model.addVar(lb=0, ub=GRB.INFINITY, name="y1")   # yield field 1
y2 = model.addVar(lb=0, ub=GRB.INFINITY, name="y2")   # yield field 2
delta2 = model.addVar(lb=0, ub=GRB.INFINITY, name="delta2")  # yield reduction

# ----------------------------
# 5. Auxiliary variables for sqrt and quadratic terms
# ----------------------------
# sqrt(x1) and sqrt(x2)
t1 = model.addVar(lb=0, ub=GRB.INFINITY, name="t1")  # t1 = sqrt(x1)
t2 = model.addVar(lb=0, ub=GRB.INFINITY, name="t2")  # t2 = sqrt(x2)

# x2² term for reduction
x2_sq = model.addVar(lb=0, ub=GRB.INFINITY, name="x2_sq")  # x2_sq = x2²

# ----------------------------
# 6. Objective
# ----------------------------
model.setObjective(y1 + y2 - delta2, GRB.MAXIMIZE)

# ----------------------------
# 7. Constraints
# ----------------------------
# Total fertilizer
model.addConstr(x1 + x2 <= total_fertilizer, "TotalFertilizer")

# Yield functions via sqrt
model.addGenConstrPow(x1, t1, 0.5, "sqrt_x1")
model.addConstr(y1 == coeff_f1 * t1, "yield_field1")

model.addGenConstrPow(x2, t2, 0.5, "sqrt_x2")
model.addConstr(y2 == coeff_f2 * t2, "yield_field2")

# Quadratic term for reduction: x2_sq = x2²
model.addGenConstrPow(x2, x2_sq, 2, "square_x2")

# Yield reduction: delta2 = 0.00002 * x2²
model.addConstr(delta2 == reduction_coeff * x2_sq, "yield_reduction")

# ----------------------------
# 8. Solve and output
# ----------------------------
model.optimize()

if model.status == GRB.OPTIMAL:
    print("Optimal solution found")
    print(f"x1 (fertilizer field 1) = {x1.X:.4f} kg")
    print(f"x2 (fertilizer field 2) = {x2.X:.4f} kg")
    print(f"y1 (yield field 1) = {y1.X:.4f} tons")
    print(f"y2 (yield field 2) = {y2.X:.4f} tons")
    print(f"delta2 (yield reduction) = {delta2.X:.6f} tons")
    print(f"Total net yield (y1+y2-delta2) = {model.objVal:.4f} tons")
    # Final answer as required
    print(f"FinalAnswer=【{model.objVal:.4f}】")
else:
    print(f"No optimal solution found. Status: {model.status}")
    print(f"FinalAnswer=【0】")