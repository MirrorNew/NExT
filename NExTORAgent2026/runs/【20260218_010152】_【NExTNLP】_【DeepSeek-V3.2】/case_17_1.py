import gurobipy as gp
from gurobipy import GRB

# 2. Define all parameter matrices and data inputs.
load_demand = 800
a = [5000, 3000, 1000]
b = [5, 3, 8]
c = [0.01, 0.02, 0.005]
P_min = [100, 50, 50]
P_max = [500, 300, 300]

# 3. Create decision variables.
model = gp.Model("PowerDispatch")
P1 = model.addVar(lb=P_min[0], ub=P_max[0], name="P1")
P2 = model.addVar(lb=P_min[1], ub=P_max[1], name="P2")
P3 = model.addVar(lb=P_min[2], ub=P_max[2], name="P3")

# 4. Create auxiliary substitution variables for quadratic terms.
# These auxiliary variables represent the squared terms: Y_i = P_i^2
Y1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Y1")
Y2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Y2")
Y3 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Y3")

# Set non-convex parameter to handle quadratic terms.
model.Params.NonConvex = 2

# Add constraints for the quadratic relationships.
model.addGenConstrPow(P1, Y1, 2, "quad1")
model.addGenConstrPow(P2, Y2, 2, "quad2")
model.addGenConstrPow(P3, Y3, 2, "quad3")

# 5. Set up the objective function.
# Minimize total fuel cost: F = sum_i (a_i + b_i * P_i + c_i * P_i^2)
obj_expr = (a[0] + b[0] * P1 + c[0] * Y1 +
            a[1] + b[1] * P2 + c[1] * Y2 +
            a[2] + b[2] * P3 + c[2] * Y3)
model.setObjective(obj_expr, GRB.MINIMIZE)

# 6. Add all constraints.
# Load balance constraint
model.addConstr(P1 + P2 + P3 == load_demand, "load_balance")

# Unit output bounds are already set in variable definitions, but add explicit constraints for clarity.
model.addConstr(P1 >= P_min[0], "P1_min")
model.addConstr(P1 <= P_max[0], "P1_max")
model.addConstr(P2 >= P_min[1], "P2_min")
model.addConstr(P2 <= P_max[1], "P2_max")
model.addConstr(P3 >= P_min[2], "P3_min")
model.addConstr(P3 <= P_max[2], "P3_max")

# 7. Solve the model and print results.
model.optimize()

# Check solution status and print results.
if model.status == GRB.OPTIMAL:
    P1_val = P1.x
    P2_val = P2.x
    P3_val = P3.x
    total_cost = model.objVal
    
    print("Optimal solution found:")
    print(f"P1 = {P1_val:.2f} MW")
    print(f"P2 = {P2_val:.2f} MW")
    print(f"P3 = {P3_val:.2f} MW")
    print(f"Total fuel cost = {total_cost:.2f} yuan")
    
    # Verify load balance
    print(f"Total generation = {P1_val + P2_val + P3_val:.2f} MW")
    print(f"Load demand = {load_demand} MW")
    
    # Output the answer to the question: the minimum total fuel cost
    print(f"FinalAnswer=【{total_cost:.2f}】")
else:
    print(f"No optimal solution found. Status: {model.status}")
    print(f"FinalAnswer=【No optimal solution found】")