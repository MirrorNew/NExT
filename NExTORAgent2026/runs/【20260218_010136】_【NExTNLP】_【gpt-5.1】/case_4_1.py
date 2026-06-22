import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Define parameters
# =========================

# Parameters List (must use exactly these values)
x2_proportion_range = [0.3, 0.6]

demand_parameters = {
    'x1': {'intercept': 5000, 'price_coef': 7},
    'x2': {'intercept': 1000, 'price_coef': 10}
}

equipment_hours = {
    'Lathe I': {'Equipment A': 3, 'Equipment B': 2, 'Equipment C': 15},
    'Lathe II': {'Equipment A': 4, 'Equipment B': 1, 'Equipment C': 2}
}

available_hours = {
    'Equipment A': 1600,
    'Equipment B': 600,
    'Equipment C': 750
}

# Helpful aliases
alpha_min, alpha_max = x2_proportion_range
x1_intercept = demand_parameters['x1']['intercept']
x1_price_coef = demand_parameters['x1']['price_coef']
x2_intercept = demand_parameters['x2']['intercept']
x2_price_coef = demand_parameters['x2']['price_coef']

# =========================
# 2. Create model
# =========================
model = gp.Model("Xies_Company_Lathe_Pricing_Production")

# Gurobi needs NonConvex = 2 for bilinear objective terms
model.Params.NonConvex = 2

# =========================
# 3. Decision variables
# =========================

# x1, x2 are integer production quantities
x1 = model.addVar(vtype=GRB.INTEGER, lb=0, ub=x1_intercept, name="x1")
x2 = model.addVar(vtype=GRB.INTEGER, lb=0, ub=x2_intercept, name="x2")

# P1, P2 are nonnegative prices
P1 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="P1")
P2 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="P2")

# =========================
# 4. Auxiliary variables (none required by the math model)
#    Placeholders, if needed later, must have lb=-GRB.INFINITY, ub=GRB.INFINITY
# =========================
# (No additional auxiliary variables are necessary for this linear-constraint, bilinear-objective model.)

# =========================
# 5. Objective function
#    Maximize total expected sales revenue: Z = P1 * x1 + P2 * x2
# =========================
model.setObjective(P1 * x1 + P2 * x2, GRB.MAXIMIZE)

# =========================
# 6. Constraints
# =========================

# 6.1 Equipment capacity constraints
# Equipment A: 3*x1 + 4*x2 <= 1600
model.addConstr(
    equipment_hours['Lathe I']['Equipment A'] * x1 +
    equipment_hours['Lathe II']['Equipment A'] * x2
    <= available_hours['Equipment A'],
    name="Equipment_A_capacity"
)

# Equipment B: 2*x1 + 1*x2 <= 600
model.addConstr(
    equipment_hours['Lathe I']['Equipment B'] * x1 +
    equipment_hours['Lathe II']['Equipment B'] * x2
    <= available_hours['Equipment B'],
    name="Equipment_B_capacity"
)

# Equipment C: 15*x1 + 2*x2 <= 750
model.addConstr(
    equipment_hours['Lathe I']['Equipment C'] * x1 +
    equipment_hours['Lathe II']['Equipment C'] * x2
    <= available_hours['Equipment C'],
    name="Equipment_C_capacity"
)

# 6.2 Product II share constraints
# x2 >= alpha_min * (x1 + x2)
model.addConstr(
    x2 >= alpha_min * (x1 + x2),
    name="Product_II_share_lower"
)

# x2 <= alpha_max * (x1 + x2)
model.addConstr(
    x2 <= alpha_max * (x1 + x2),
    name="Product_II_share_upper"
)

# 6.3 Demand–price relationships
# x1 + 7*P1 = 5000
model.addConstr(
    x1 + x1_price_coef * P1 == x1_intercept,
    name="Demand_function_I"
)

# x2 + 10*P2 = 1000
model.addConstr(
    x2 + x2_price_coef * P2 == x2_intercept,
    name="Demand_function_II"
)

# =========================
# 7. Solve the model
# =========================
model.optimize()

# =========================
# 8. Print results
# =========================
if model.status == GRB.OPTIMAL:
    x1_val = x1.X
    x2_val = x2.X
    P1_val = P1.X
    P2_val = P2.X
    Z_val = model.ObjVal

    print("Optimal solution found:")
    print(f"x1 (Lathe I quantity)  = {x1_val}")
    print(f"x2 (Lathe II quantity) = {x2_val}")
    print(f"P1 (Lathe I price)     = {P1_val}")
    print(f"P2 (Lathe II price)    = {P2_val}")
    print(f"Maximum total revenue  = {Z_val}")

    # The question asks: "How should the company balance demand and price to maximize the expected total sales price?"
    # We return the maximum total sales revenue as the final numerical answer.
    print(f"FinalAnswer=【{Z_val}】")
else:
    print(f"Optimization ended with status {model.status}")
    # If no optimal solution is found, report NaN as final answer
    print("FinalAnswer=【nan】")