import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
# Parameters List
load_demand_MW = 100.0
loss_coefficient_P1 = 0.0005
loss_coefficient_P2 = 0.001
cost_coeff_P1_linear = 5.0
cost_coeff_P1_quadratic = 0.02
cost_coeff_P2_linear = 4.0
cost_coeff_P2_quadratic = 0.025

# Create Gurobi model
model = gp.Model("Economic_Load_Dispatch")

# Set NonConvex parameter to 2 to handle quadratic equality constraints
model.Params.NonConvex = 2

# 2. Create decision variables.
P1 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="P1")
P2 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="P2")
L1 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="L1")
L2 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="L2")
C1 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="C1")
C2 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="C2")

# 3. Create auxiliary substitution variables for quadratic terms
# P1_sq = P1^2
P1_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P1_sq")
# P2_sq = P2^2
P2_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P2_sq")

# 4. Set up the objective function.
# Minimize total cost Z = C1 + C2
model.setObjective(C1 + C2, GRB.MINIMIZE)

# 5. Add all constraints.

# Auxiliary constraints for squares using General Constraints
# P1_sq = P1^2
model.addGenConstrPow(P1, P1_sq, 2, name="Link_P1_sq")
# P2_sq = P2^2
model.addGenConstrPow(P2, P2_sq, 2, name="Link_P2_sq")

# Line Loss Definitions
# L1 = 0.0005 * P1^2
model.addConstr(L1 == loss_coefficient_P1 * P1_sq, name="Loss1_definition")
# L2 = 0.0010 * P2^2
model.addConstr(L2 == loss_coefficient_P2 * P2_sq, name="Loss2_definition")

# Generation Cost Definitions
# C1 = 5*P1 + 0.02*P1^2
model.addConstr(C1 == cost_coeff_P1_linear * P1 + cost_coeff_P1_quadratic * P1_sq, name="Cost1_definition")
# C2 = 4*P2 + 0.025*P2^2
model.addConstr(C2 == cost_coeff_P2_linear * P2 + cost_coeff_P2_quadratic * P2_sq, name="Cost2_definition")

# Power Balance Constraint
# P1 + P2 - L1 - L2 = 100
model.addConstr(P1 + P2 - L1 - L2 == load_demand_MW, name="Power_balance")

# 6. Solve the model and print results.
model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"Optimal Total Cost: {model.ObjVal}")
    print(f"P1: {P1.X} MW")
    print(f"P2: {P2.X} MW")
    print(f"L1: {L1.X} MW")
    print(f"L2: {L2.X} MW")
    print(f"C1: {C1.X} $/h")
    print(f"C2: {C2.X} $/h")
    # Output the final answer
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimization was not successful.")