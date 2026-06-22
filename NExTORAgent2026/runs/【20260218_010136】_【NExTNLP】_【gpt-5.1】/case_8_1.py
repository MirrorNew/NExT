import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Define parameters
# =========================

# Given parameter lists (must use these values)
price_A = [70.0, -4.0]        # p1 = 70 - 4 x1
price_B = [150.0, -15.0]      # p2 = 150 - 15 x2
unit_production_cost = 150000  # yuan per unit
machine_loss_fee = 1000000     # yuan total

# Convert costs to units of 10,000 yuan to match price units
unit_cost_10k = unit_production_cost / 10000.0   # 150000 / 10000 = 15
fixed_cost_10k = machine_loss_fee / 10000.0      # 1000000 / 10000 = 100

# =========================
# 2. Create model
# =========================
model = gp.Model("TwoCustomerPricing")

# Allow non-convex quadratic expressions (we are maximizing a concave quadratic)
model.Params.NonConvex = 2

# =========================
# 3. Decision variables
# =========================
# x1, x2: quantities to A and B (units)
x1 = model.addVar(lb=0.0, name="x1")
x2 = model.addVar(lb=0.0, name="x2")

# p1, p2: prices (in 10,000 yuan)
p1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="p1")
p2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="p2")

# pi: total profit (in 10,000 yuan)
pi = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="pi")

# =========================
# 4. Auxiliary / substitution variables (none beyond pi, p1, p2)
#    (All auxiliary vars already created with full range)
# =========================

# =========================
# 5. Constraints
# =========================

# Price–demand relationships
# p1 = 70 - 4 x1
model.addConstr(
    p1 == price_A[0] + price_A[1] * x1,
    name="PriceFunc_A"
)

# p2 = 150 - 15 x2
model.addConstr(
    p2 == price_B[0] + price_B[1] * x2,
    name="PriceFunc_B"
)

# Profit definition (in 10,000 yuan):
# pi = x1*p1 + x2*p2 - unit_cost_10k*(x1 + x2) - fixed_cost_10k
model.addConstr(
    pi == x1 * p1 + x2 * p2 - unit_cost_10k * (x1 + x2) - fixed_cost_10k,
    name="ProfitDef"
)

# Non-negativity of x1, x2 already enforced by variable bounds

# =========================
# 6. Objective
# =========================
model.setObjective(pi, GRB.MAXIMIZE)

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
    p1_opt = p1.X
    p2_opt = p2.X
    pi_opt_10k = pi.X  # profit in 10,000 yuan

    print(f"Optimal quantity to Customer A (x1): {x1_opt:.6f} units")
    print(f"Optimal quantity to Customer B (x2): {x2_opt:.6f} units")
    print(f"Optimal price to Customer A (p1): {p1_opt:.6f} (10,000 yuan)")
    print(f"Optimal price to Customer B (p2): {p2_opt:.6f} (10,000 yuan)")
    print(f"Maximum profit: {pi_opt_10k:.6f} (10,000 yuan)")

    # FinalAnswer: maximum profit in 10,000 yuan (as required)
    print(f"FinalAnswer=【{pi_opt_10k}】")
else:
    # In case no optimal solution is found, still print something
    print("No optimal solution found.")
    print("FinalAnswer=【None】")