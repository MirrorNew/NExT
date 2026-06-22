import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
total_water_resources = 100
number_of_farms = 3
a = [5, 3, 4]
total_irrigation_amount_limit = 100
sum_of_squares_of_irrigation_water_limit = 3500

# 2. Create Gurobi Model
model = gp.Model("Irrigation_Optimization")

# Set parameter to handle non-convex/nonlinear constraints (required for GenConstrPow)
model.Params.NonConvex = 2

# 3. Create decision variables
# w_j: irrigation water allocated to farm j (j=0..2 in Python, corresponding to 1..3 in problem)
w = model.addVars(number_of_farms, lb=0.0, name="w")

# 4. Create auxiliary substitution variables
# y_j: to store the value of w_j^(exponents)
# As per instructions, range is set to -inf to +inf, though practically non-negative here.
y = model.addVars(number_of_farms, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y")

# 5. Set up the objective function
# Objective: Maximize Z = sum(a_j * w_j^(j/4))
# Note: j in problem is 1-based (1, 2, 3). In Python, indices are 0, 1, 2.
# Term j=1 (index 0): a[0] * w[0]^(1/4)
# Term j=2 (index 1): a[1] * w[1]^(2/4)
# Term j=3 (index 2): a[2] * w[2]^(3/4)
obj_expr = gp.quicksum(a[j] * y[j] for j in range(number_of_farms))
model.setObjective(obj_expr, GRB.MAXIMIZE)

# 6. Add all constraints

# Constraint: Power function relations for auxiliary variables
# w_j in problem maps to w[j] in code. Exponent is (j+1)/4.
for j in range(number_of_farms):
    exponent = (j + 1) / 4.0
    # y[j] = w[j] ^ exponent
    model.addGenConstrPow(w[j], y[j], exponent, name=f"Pow_w{j}")

# Constraint: Total water availability
# sum(w_j) <= 100
model.addConstr(gp.quicksum(w[j] for j in range(number_of_farms)) <= total_irrigation_amount_limit, "TotalWaterLimit")

# Constraint: Quadratic water-usage limit
# sum(w_j^2) <= 3500
model.addQConstr(gp.quicksum(w[j] * w[j] for j in range(number_of_farms)) <= sum_of_squares_of_irrigation_water_limit, "SumSquaresLimit")

# 7. Solve the model and print results
model.optimize()

if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    for j in range(number_of_farms):
        print(f"Farm {j+1} (w_{j+1}): {w[j].X:.4f}")
    print(f"Total Yield: {model.objVal:.4f}")
    # Output the final answer in the required format
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found.")