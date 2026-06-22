import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
# Parameters from the Parameters List
delivery_quantities = [40, 60, 80]
max_capacity = 100
cost_function_coeffs = {'a': 50, 'b': 0.002, 'exp': 2.9}
storage_cost = 4
initial_inventory = 0

# Number of periods (quarters)
T = len(delivery_quantities)

# 2. Create Gurobi Model
model = gp.Model("Engine_Production_Optimization")

# Set parameter to allow non-convex quadratic/nonlinear constraints (required for x^2.9)
model.Params.NonConvex = 2

# 3. Create decision variables
# x[t]: Number of engines produced in quarter t (t=0,1,2 corresponding to Q1,Q2,Q3)
x = model.addVars(T, vtype=GRB.INTEGER, lb=0, ub=max_capacity, name="x")

# I[t]: Inventory at the end of quarter t
# I_t can theoretically be larger than capacity if accumulation occurs, but is bounded by total demand in practice
I = model.addVars(T, vtype=GRB.INTEGER, lb=0, name="I")

# 4. Create auxiliary substitution variables
# y[t] = x[t]^2.9. Since x is integer, y will be continuous.
# y represents the nonlinear part of the production cost.
y = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="y")

# Add General Constraints for Power Function: y[t] = x[t] ^ 2.9
# Note: Syntax is model.addGenConstrPow(xvar, yvar, exponent) => y = x^a
for t in range(T):
    model.addGenConstrPow(x[t], y[t], cost_function_coeffs['exp'], name=f"pow_x_{t}")

# 5. Set up the objective function
# Minimize Total Cost = Sum of (Production Cost + Storage Cost)
# Production Cost per quarter = 50*x_t + 0.002*x_t^2.9
# Storage Cost per quarter = 4*I_t
# Using auxiliary variable y_t = x_t^2.9
obj_expr = gp.quicksum(
    cost_function_coeffs['a'] * x[t] + 
    cost_function_coeffs['b'] * y[t] + 
    storage_cost * I[t]
    for t in range(T)
)
model.setObjective(obj_expr, GRB.MINIMIZE)

# 6. Add all constraints
# Inventory Balance Constraints: I_t = I_{t-1} + x_t - D_t
for t in range(T):
    if t == 0:
        # For the first quarter, previous inventory is initial_inventory
        model.addConstr(I[t] == initial_inventory + x[t] - delivery_quantities[t], name=f"InventoryBalance_{t}")
    else:
        # For subsequent quarters
        model.addConstr(I[t] == I[t-1] + x[t] - delivery_quantities[t], name=f"InventoryBalance_{t}")

# 7. Solve the model and print results
model.optimize()

if model.Status == GRB.OPTIMAL:
    print("\nOptimal Solution Found:")
    for t in range(T):
        print(f"Quarter {t+1}: Produce {x[t].X:.0f} units, Inventory {I[t].X:.0f} units, NonlinearCostTerm {y[t].X:.4f}")
    
    # Output the final answer in the required format
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimization was not successful.")