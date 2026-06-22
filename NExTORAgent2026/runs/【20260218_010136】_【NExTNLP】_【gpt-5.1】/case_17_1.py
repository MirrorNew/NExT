import gurobipy as gp
from gurobipy import GRB

# ==============================
# 1. Define parameters (from Parameters List)
# ==============================
load_demand = 800
a = [5000, 3000, 1000]
b = [5, 3, 8]
c = [0.01, 0.02, 0.005]
P_min = [100, 50, 50]
P_max = [500, 300, 300]

# ==============================
# 2. Create model
# ==============================
model = gp.Model("Economic_Dispatch_3_Units")

# Allow general constraints with powers (as required by instructions)
model.Params.NonConvex = 2

# ==============================
# 3. Create decision variables P1, P2, P3
# ==============================
# Index mapping: 0 -> Unit 1, 1 -> Unit 2, 2 -> Unit 3
P = model.addVars(3,
                  lb=[P_min[i] for i in range(3)],
                  ub=[P_max[i] for i in range(3)],
                  vtype=GRB.CONTINUOUS,
                  name="P")

# ==============================
# 4. Create auxiliary substitution variables for P_i^2 (Q1, Q2, Q3)
#    Range must be (-inf, +inf) as requested
# ==============================
Q = model.addVars(3,
                  lb=-GRB.INFINITY,
                  ub=GRB.INFINITY,
                  vtype=GRB.CONTINUOUS,
                  name="Q")

# Define Q[i] = P[i]^2 via general power constraints
for i in range(3):
    model.addGenConstrPow(P[i], Q[i], 2.0, name=f"Pow_P{i+1}")

# ==============================
# 5. Load balance constraint
# ==============================
model.addConstr(P[0] + P[1] + P[2] == load_demand, name="LoadBalance")

# (Bounds are already included in variable definitions; no extra constraints needed,
# but could be added explicitly if desired.)

# ==============================
# 6. Objective: Minimize total fuel cost
#     F = sum_i (a_i + b_i * P_i + c_i * P_i^2)
#         = sum_i a_i + sum_i b_i * P_i + sum_i c_i * Q_i
# ==============================
constant_term = sum(a)
linear_part = gp.quicksum(b[i] * P[i] for i in range(3))
quad_part = gp.quicksum(c[i] * Q[i] for i in range(3))

model.setObjective(constant_term + linear_part + quad_part, GRB.MINIMIZE)

# ==============================
# 7. Solve the model
# ==============================
model.optimize()

# ==============================
# 8. Print results
# ==============================
if model.status == GRB.OPTIMAL:
    P1_opt = P[0].X
    P2_opt = P[1].X
    P3_opt = P[2].X
    total_cost = model.ObjVal

    print("Optimal dispatch solution:")
    print(f"P1 = {P1_opt:.4f} MW")
    print(f"P2 = {P2_opt:.4f} MW")
    print(f"P3 = {P3_opt:.4f} MW")
    print(f"Total fuel cost = {total_cost:.4f} yuan")
else:
    print("No optimal solution found.")
    P1_opt = float('nan')
    P2_opt = float('nan')
    P3_opt = float('nan')
    total_cost = float('nan')

# ==============================
# 9. FinalAnswer output (as required)
#    The question asks for the lowest total fuel cost,
#    so FinalAnswer is the optimal objective value.
# ==============================
print(f"FinalAnswer=【{total_cost}】")