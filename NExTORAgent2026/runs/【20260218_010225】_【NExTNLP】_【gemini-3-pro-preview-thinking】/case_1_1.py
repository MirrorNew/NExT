import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
# Parameters defined in the problem
b = 200              # Production limit (units/month)
c = 100              # Storage limit (units)
y_0 = 0              # Initial storage
y_3_req = 0          # Required storage at end of quarter
d = [150, 200, 212]  # Demand for months 1, 2, 3

# Production cost function parameters
# f_1(x) = 100 * x^0.9
# f_2(x) = 100 * x^0.8
# f_3(x) = 150 * x^0.5
f_coeffs = [100, 100, 150]
f_powers = [0.9, 0.8, 0.5]

# Storage cost coefficient
g_coeff = 20

# 2. Create the Gurobi model.
model = gp.Model("HuaxinElectronics_ProductionPlan")

# Important: The objective involves terms like x^0.9, x^0.8. Since 0 < power < 1, 
# these are concave functions. Minimizing a concave function is a non-convex global 
# optimization problem. We must set NonConvex=2 to allow Gurobi to solve it.
model.Params.NonConvex = 2

# 3. Create decision variables.
# x_i: Number of TV sets produced in month i (i=1..3)
# 0 <= x_i <= b
x = model.addVars(3, lb=0, ub=b, vtype=GRB.CONTINUOUS, name="x")

# y_i: Number of TV sets in inventory at end of month i (i=1..3)
# 0 <= y_i <= c
y = model.addVars(3, lb=0, ub=c, vtype=GRB.CONTINUOUS, name="y")

# 4. Create auxiliary substitution variables.
# u_i will represent the nonlinear term x_i^(power)
# These variables can range from 0 to infinity (though effectively bounded by x's bounds)
u = model.addVars(3, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="u")

# Add generalized constraints for power functions: u[i] = x[i] ^ f_powers[i]
# Note: In addGenConstrPow(x, y, a), the relation is y = x^a.
for i in range(3):
    model.addGenConstrPow(x[i], u[i], f_powers[i], name=f"power_constr_{i+1}")

# 5. Set up the objective function.
# Minimize Z = Sum(f_coeff * u_i) + Sum(g_coeff * y_i)
production_cost = gp.quicksum(f_coeffs[i] * u[i] for i in range(3))
storage_cost = gp.quicksum(g_coeff * y[i] for i in range(3))

model.setObjective(production_cost + storage_cost, GRB.MINIMIZE)

# 6. Add all constraints.

# Inventory Balance Constraints: y_{i-1} + x_i - d_i = y_i
# i = 1 (index 0): y_0 + x_1 - d_1 = y_1
model.addConstr(y_0 + x[0] - d[0] == y[0], name="Balance_Month_1")

# i = 2 (index 1): y_1 + x_2 - d_2 = y_2
model.addConstr(y[0] + x[1] - d[1] == y[1], name="Balance_Month_2")

# i = 3 (index 2): y_2 + x_3 - d_3 = y_3
model.addConstr(y[1] + x[2] - d[2] == y[2], name="Balance_Month_3")

# Ending Inventory Constraint: y_3 = 0
model.addConstr(y[2] == y_3_req, name="End_Inventory_Zero")

# 7. Solve the model and print results.
model.optimize()

if model.Status == GRB.OPTIMAL:
    print(f"\nOptimal Objective Value: {model.ObjVal}")
    for i in range(3):
        print(f"Month {i+1}: Produced {x[i].X:.2f}, Stored {y[i].X:.2f}")
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimization was not successful.")