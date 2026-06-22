import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
E = [100, 150, 80]
alpha = [0.5, 0.8, 1.0]
R_min = 120
bonus_rate = 60
num_factories = len(E)

# 2. Create model.
model = gp.Model("GreenTransformation")

# Advice: use "model.Params.NonConvex = 2" if needed for quadratic equality constraints via auxiliary variables.
model.Params.NonConvex = 2

# 3. Create decision variables.
# x[i]: Emission reduction of factory i (tons), 0 <= x_i <= E_i
x = model.addVars(num_factories, lb=0.0, ub=E, vtype=GRB.CONTINUOUS, name="x")

# B: Total bonus awarded (10,000 yuan), B >= 0
B = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="B")

# 4. Create any auxiliary substitution or indicator variables.
# y[i] to substitute x[i]^2. Bounds set to infinity as per instructions for auxiliary vars.
y = model.addVars(num_factories, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y")

# 5. Set up the objective function.
# Minimize Total Cost = Sum(alpha_i * x_i^2) - B
# Substitute x_i^2 with y_i
obj_expr = gp.quicksum(alpha[i] * y[i] for i in range(num_factories)) - B
model.setObjective(obj_expr, GRB.MINIMIZE)

# 6. Add all constraints.

# Constraint: Total reduction requirement
# sum(x_i) >= 120
total_reduction = gp.quicksum(x[i] for i in range(num_factories))
model.addConstr(total_reduction >= R_min, name="TotalReductionReq")

# Constraint: Bonus definition
# B = 60 * (sum(x_i) - 120)
# Note: Since sum(x_i) >= 120 constraint exists, the term (sum(x_i) - 120) is always non-negative.
# Thus, B = 60 * max(sum(x_i) - 120, 0) simplifies to linear equality.
model.addConstr(B == bonus_rate * (total_reduction - R_min), name="BonusDefinition")

# Constraint: Auxiliary substitution y_i = x_i^2
# Using General Constraint Power: y = x^2
for i in range(num_factories):
    model.addGenConstrPow(x[i], y[i], 2, name=f"PowConstraint_{i}")

# 7. Solve the model and print results.
model.optimize()

if model.Status == GRB.OPTIMAL:
    print("Optimization Successful.")
    print(f"Objective Value (Minimum Total Cost): {model.ObjVal}")
    for i in range(num_factories):
        print(f"Factory {i+1} Reduction (x_{i+1}): {x[i].X:.4f}")
    print(f"Total Bonus (B): {B.X:.4f}")
    
    # Final Answer output as requested
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("No optimal solution found.")