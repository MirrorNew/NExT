import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
load_demand = 800
a = [5000, 3000, 1000]
b = [5, 3, 8]
c = [0.01, 0.02, 0.005]
P_min = [100, 50, 50]
P_max = [500, 300, 300]

# 2. Create the model.
model = gp.Model("PowerDispatchOptimization")
model.Params.NonConvex = 2

# 3. Create decision variables.
P1 = model.addVar(lb=P_min[0], ub=P_max[0], vtype=GRB.CONTINUOUS, name="P1")
P2 = model.addVar(lb=P_min[1], ub=P_max[1], vtype=GRB.CONTINUOUS, name="P2")
P3 = model.addVar(lb=P_min[2], ub=P_max[2], vtype=GRB.CONTINUOUS, name="P3")

# 4. Create auxiliary substitution variables.
P1_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P1_sq")
P2_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P2_sq")
P3_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P3_sq")

# 5. Add general constraints for the squared terms.
model.addGenConstrPow(P1, P1_sq, 2)
model.addGenConstrPow(P2, P2_sq, 2)
model.addGenConstrPow(P3, P3_sq, 2)

# 6. Set up the objective function.
# F = (a1 + b1*P1 + c1*P1^2) + (a2 + b2*P2 + c2*P2^2) + (a3 + b3*P3 + c3*P3^2)
objective = (a[0] + b[0] * P1 + c[0] * P1_sq) + \
            (a[1] + b[1] * P2 + c[1] * P2_sq) + \
            (a[2] + b[2] * P3 + c[2] * P3_sq)
model.setObjective(objective, GRB.MINIMIZE)

# 7. Add constraints.
# Load balance constraint: P1 + P2 + P3 = 800
model.addConstr(P1 + P2 + P3 == load_demand, "LoadBalance")

# 8. Solve the model and print results.
model.optimize()

if model.Status == GRB.OPTIMAL:
    total_cost = model.ObjVal
    print(f"Optimal P1: {P1.X} MW")
    print(f"Optimal P2: {P2.X} MW")
    print(f"Optimal P3: {P3.X} MW")
    print(f"Total Fuel Cost: {total_cost} yuan")
    print(f"FinalAnswer=【{total_cost}】")
else:
    print("Optimization was not successful.")