import gurobipy as gp
from gurobipy import GRB

# Create model
model = gp.Model("ELD_NonlinearLoss")

# Parameters from the list
load_demand = 100.0
loss_coeff_P1 = 0.0005
loss_coeff_P2 = 0.001
cost_coeff_P1_lin = 5.0
cost_coeff_P1_quad = 0.02
cost_coeff_P2_lin = 4.0
cost_coeff_P2_quad = 0.025

# Decision variables
P1 = model.addVar(lb=0, name="P1")
P2 = model.addVar(lb=0, name="P2")
L1 = model.addVar(lb=0, name="L1")
L2 = model.addVar(lb=0, name="L2")
C1 = model.addVar(lb=0, name="C1")
C2 = model.addVar(lb=0, name="C2")

# Auxiliary substitution variables for squares
P1_sq = model.addVar(lb=0, name="P1_sq")
P2_sq = model.addVar(lb=0, name="P2_sq")

# Set non-convex parameter for quadratic constraints
model.Params.NonConvex = 2

# Add constraints for squares using general constraints
model.addGenConstrPow(P1, P1_sq, 2, "P1_squared")
model.addGenConstrPow(P2, P2_sq, 2, "P2_squared")

# Power balance constraint
model.addConstr(P1 + P2 - L1 - L2 == load_demand, "Power_balance")

# Loss definition constraints
model.addConstr(L1 == loss_coeff_P1 * P1_sq, "Loss1_definition")
model.addConstr(L2 == loss_coeff_P2 * P2_sq, "Loss2_definition")

# Cost definition constraints
model.addConstr(C1 == cost_coeff_P1_lin * P1 + cost_coeff_P1_quad * P1_sq, "Cost1_definition")
model.addConstr(C2 == cost_coeff_P2_lin * P2 + cost_coeff_P2_quad * P2_sq, "Cost2_definition")

# Set objective: minimize total cost
model.setObjective(C1 + C2, GRB.MINIMIZE)

# Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"P1 = {P1.X:.2f} MW")
    print(f"P2 = {P2.X:.2f} MW")
    print(f"L1 = {L1.X:.4f} MW")
    print(f"L2 = {L2.X:.4f} MW")
    print(f"C1 = ${C1.X:.2f}/h")
    print(f"C2 = ${C2.X:.2f}/h")
    print(f"Total cost = ${model.ObjVal:.2f}/h")
    
    # Calculate the answer: total generation cost (objective value)
    total_cost = model.ObjVal
    print(f"FinalAnswer=【{total_cost:.2f}】")
else:
    print(f"Optimization failed with status {model.status}")
    print(f"FinalAnswer=【Optimization failed】")