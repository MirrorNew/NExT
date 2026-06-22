import gurobipy as gp
from gurobipy import GRB

# ==============================
# 1. Define Parameters
# ==============================
number_of_products = 2
capacity = 300
space_per_unit = [None, 1, 2]  # Index 1..2 as specified
D = [800, 400]
K = [50, 50]
h = [1, 1]
beverage_space_ratio = 1.5  # Not used directly in this 2-product model

# ==============================
# 2. Create Model
# ==============================
model = gp.Model("EOQ_with_storage_capacity")

# Allow non-convex (bilinear) constraints for reciprocals
model.Params.NonConvex = 2

# ==============================
# 3. Decision Variables
# ==============================
# Q1, Q2: order quantities
Q = {}
for i in range(number_of_products):
    # Q_i >= 0; we will enforce strict positivity via constraints to avoid division by zero
    Q[i] = model.addVar(lb=0.0, name=f"Q_{i+1}")

# ==============================
# 4. Auxiliary Variables
# ==============================
# Y1 = 1 / Q1, Y2 = 1 / Q2
Y = {}
for i in range(number_of_products):
    Y[i] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"Y_{i+1}")

model.update()

# ==============================
# 5. Objective Function
# ==============================
# Z = 50*(800/Q1) + Q1/2 + 50*(400/Q2) + Q2/2
# Using substitution: 800/Q1 = 800 * Y1, 400/Q2 = 400 * Y2
obj = (
    K[0] * D[0] * Y[0] + 0.5 * Q[0] +
    K[1] * D[1] * Y[1] + 0.5 * Q[1]
)
model.setObjective(obj, GRB.MINIMIZE)

# ==============================
# 6. Constraints
# ==============================

# 6.1 Reciprocal link constraints: Q_i * Y_i = 1  (for i = 1,2)
for i in range(number_of_products):
    model.addConstr(Q[i] * Y[i] == 1, name=f"reciprocal_Q_{i+1}")

# 6.2 Storage capacity constraint:
# (1*(Q1/2) + 2*(Q2/2)) ≤ 300  => 0.5*Q1 + Q2 <= 300
model.addConstr(0.5 * Q[0] + Q[1] <= capacity, name="storage_capacity")

# 6.3 Positivity constraints to avoid division by zero in EOQ model
model.addConstr(Q[0] >= 1e-6, name="Q1_pos")
model.addConstr(Q[1] >= 1e-6, name="Q2_pos")

# ==============================
# 7. Solve Model
# ==============================
model.optimize()

# ==============================
# 8. Print Results
# ==============================
if model.Status == GRB.OPTIMAL:
    Q1_opt = Q[0].X
    Q2_opt = Q[1].X
    Z_opt = model.ObjVal

    print(f"Optimal order quantity Q1: {Q1_opt:.6f}")
    print(f"Optimal order quantity Q2: {Q2_opt:.6f}")
    print(f"Minimum total annual cost Z: {Z_opt:.6f}")
else:
    print("No optimal solution found.")
    Q1_opt = float('nan')
    Q2_opt = float('nan')
    Z_opt = float('nan')

# ==============================
# 9. Final Answer Output
# ==============================
# The question asks: "Determine the order quantities Q1, Q2"
# We encode them together in the final answer string.
print(f"FinalAnswer=【Q1={Q1_opt:.6f}, Q2={Q2_opt:.6f}】")