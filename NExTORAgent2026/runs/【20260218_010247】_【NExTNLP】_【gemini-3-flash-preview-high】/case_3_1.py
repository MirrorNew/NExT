import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and any other necessary packages.

# 2. Define all parameter matrices and data inputs.
delivery_quantities = [40, 60, 80]
max_capacity = 100
cost_function_coeffs = {'a': 50, 'b': 0.002, 'exp': 2.9}
storage_cost = 4
initial_inventory = 0
num_quarters = len(delivery_quantities)

# 3. Create a model
model = gp.Model("EngineProductionOptimization")

# 4. Create decision variables.
# Number of engines produced in quarter t (integer, 0 to max_capacity)
x = model.addVars(num_quarters, lb=0, ub=max_capacity, vtype=GRB.INTEGER, name="x")
# Inventory at the end of quarter t (integer, non-negative)
I = model.addVars(num_quarters, lb=0, vtype=GRB.INTEGER, name="I")

# 5. Create any auxiliary substitution variables.
# y_t = x_t^2.9 (continuous)
y = model.addVars(num_quarters, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y")

# Set up parameters for general constraints
model.Params.NonConvex = 2

# 6. Set up the objective function.
# Minimize total cost = Σ_{t=1}^3 [50*x_t + 0.002*x_t^2.9] + 4*Σ_{t=1}^3 I_t
# Using auxiliary variable y_t: Minimize Σ_{t=1}^3 [a*x_t + b*y_t + storage_cost*I_t]
objective = gp.quicksum(cost_function_coeffs['a'] * x[t] + 
                        cost_function_coeffs['b'] * y[t] + 
                        storage_cost * I[t] for t in range(num_quarters))
model.setObjective(objective, GRB.MINIMIZE)

# 7. Add all constraints.
for t in range(num_quarters):
    # Inventory Balance constraints: I_t = I_{t-1} + x_t - D_t
    if t == 0:
        model.addConstr(I[t] == initial_inventory + x[t] - delivery_quantities[t], name=f"InvBalance_{t}")
    else:
        model.addConstr(I[t] == I[t-1] + x[t] - delivery_quantities[t], name=f"InvBalance_{t}")
    
    # Power function constraint: y_t = x_t^2.9
    # addGenConstrPow(xvar, yvar, a, options="") represents yvar = xvar^a
    model.addGenConstrPow(x[t], y[t], cost_function_coeffs['exp'], name=f"PowerConstr_{t}")

# 8. Solve the model and print results.
model.optimize()

if model.status == GRB.OPTIMAL:
    print("Optimal Solution Found:")
    for t in range(num_quarters):
        print(f"Quarter {t+1}: Produced = {x[t].X}, End Inventory = {I[t].X}")
    total_cost = model.objVal
    print(f"Total Cost: {total_cost}")
else:
    print("No optimal solution found.")
    total_cost = None

# Output the answer to the question (the objective value)
print(f"FinalAnswer=【{total_cost}】")