import gurobipy as gp
from gurobipy import GRB

# 1. Parameter definitions from Parameters List
peak_period_year = 2023
transmission_distance_threshold_km = 200
ieee_test_nodes = 14
annual_fuel_cost_saving_yuan = 120000000.0
emission_reduction_rate_percent = 0.3
n_power_sources = 2
n_load_nodes = 1
load_demand_MW = 100.0
loss_coefficient_P1 = 0.0005
loss_coefficient_P2 = 0.001
cost_coeff_P1_linear = 5.0
cost_coeff_P1_quadratic = 0.02
cost_coeff_P2_linear = 4.0
cost_coeff_P2_quadratic = 0.025

# 2. Create the model
model = gp.Model("Kazdale_Energy_Dispatch")

# Set the NonConvex parameter to 2 for nonlinear/non-convex constraints
model.Params.NonConvex = 2

# 3. Decision variables from context
P1 = model.addVar(lb=0.0, name="P1")
P2 = model.addVar(lb=0.0, name="P2")
L1 = model.addVar(lb=0.0, name="L1")
L2 = model.addVar(lb=0.0, name="L2")
C1 = model.addVar(lb=0.0, name="C1")
C2 = model.addVar(lb=0.0, name="C2")

# 4. Auxiliary substitution variables for squared terms
# As per coding advice: P1_sq = P1^2, P2_sq = P2^2
# These variables should range from negative infinity to positive infinity.
P1_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="P1_sq")
P2_sq = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="P2_sq")

# 5. Objective Function: Minimize total generation cost Z = C1 + C2
model.setObjective(C1 + C2, GRB.MINIMIZE)

# 6. Constraints
# Defining the squared terms using addGenConstrPow
model.addGenConstrPow(P1, P1_sq, 2)
model.addGenConstrPow(P2, P2_sq, 2)

# Line loss definitions using the auxiliary squared variables
# L1 = 0.0005 * P1^2
model.addConstr(L1 == loss_coefficient_P1 * P1_sq, name="Loss1_definition")
# L2 = 0.0010 * P2^2
model.addConstr(L2 == loss_coefficient_P2 * P2_sq, name="Loss2_definition")

# Power balance constraint: P1 + P2 - L1 - L2 = Load Demand
model.addConstr(P1 + P2 - L1 - L2 == load_demand_MW, name="Power_balance")

# Generation cost definitions using original and squared variables
# C1 = 5*P1 + 0.02*P1^2
model.addConstr(C1 == cost_coeff_P1_linear * P1 + cost_coeff_P1_quadratic * P1_sq, name="Cost1_definition")
# C2 = 4*P2 + 0.025*P2^2
model.addConstr(C2 == cost_coeff_P2_linear * P2 + cost_coeff_P2_quadratic * P2_sq, name="Cost2_definition")

# 7. Solve the model and print results
model.optimize()

if model.status == GRB.OPTIMAL:
    # Final answer is the minimized total cost
    print(f"FinalAnswer=【{model.ObjVal}】")