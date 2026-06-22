import gurobipy as gp
from gurobipy import GRB

# ==============================
# 1. Define parameters (from Parameters List)
# ==============================
octane_threshold = 80                  # octane_threshold
cost_reduction_range = (0.05, 0.1)     # cost_reduction_range (not used directly in model)
environmental_compliance_rate = 1.0    # environmental_compliance_rate (not used directly)
unit_price_A = 30                      # unit_price_A
quality_index_A = 90                   # quality_index_A
unit_price_B = 20                      # unit_price_B
quality_index_B = 70                   # quality_index_B
quality_index_min = 80                 # quality_index_min
total_flow = 1000                      # total_flow
flow_threshold_A = 450                 # flow_threshold_A
penalty_exponent_A = 1.05              # penalty_exponent_A

# ==============================
# 2. Create model
# ==============================
model = gp.Model("Kazdel_Refinery_Octane_Mix")

# Enable non-convex features for power constraints
model.Params.NonConvex = 2

# ==============================
# 3. Decision variables
# ==============================
# Flow rates
x_A = model.addVar(lb=0, ub=total_flow, vtype=GRB.CONTINUOUS, name="x_A")
x_B = model.addVar(lb=0, ub=total_flow, vtype=GRB.CONTINUOUS, name="x_B")

# Adjusted flow of A used in cost
f_A = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="f_A")

# Binary indicator for region x_A > flow_threshold_A
y_A = model.addVar(vtype=GRB.BINARY, name="y_A")

# ==============================
# 4. Auxiliary substitution variables
# ==============================
# p_A represents x_A^penalty_exponent_A
p_A = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="p_A")

# (Example of general aux var following instructions: unrestricted var)
aux_dummy = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                         vtype=GRB.CONTINUOUS, name="aux_dummy")

# ==============================
# 5. Objective function
# ==============================
# Minimize total raw-material cost = unit_price_A * f_A + unit_price_B * x_B
model.setObjective(unit_price_A * f_A + unit_price_B * x_B, GRB.MINIMIZE)

# ==============================
# 6. Constraints
# ==============================

# 6.1 Total flow constraint
model.addConstr(x_A + x_B == total_flow, name="TotalFlow")

# 6.2 Mixture octane constraint
# (quality_index_A * x_A + quality_index_B * x_B) / (x_A + x_B) >= quality_index_min
# With x_A + x_B = total_flow, becomes:
# quality_index_A * x_A + quality_index_B * x_B >= quality_index_min * total_flow
model.addConstr(
    quality_index_A * x_A + quality_index_B * x_B >= quality_index_min * total_flow,
    name="OctaneConstraint"
)

# 6.3 Power relation p_A = x_A^penalty_exponent_A
model.addGenConstrPow(x_A, p_A, penalty_exponent_A, name="Power_xA")

# 6.4 Piecewise definition of f_A using indicator constraints:
#     If x_A <= flow_threshold_A, then y_A = 0 region:
#         x_A <= 450, f_A = x_A
#     If x_A > flow_threshold_A, then y_A = 1 region:
#         x_A >= 451, f_A = p_A = x_A^1.05

# Region 0: y_A = 0  -->  x_A <= 450,  f_A = x_A
model.addGenConstrIndicator(y_A, 0, x_A <= flow_threshold_A, name="Region0_xA_le_450")
model.addGenConstrIndicator(y_A, 0, f_A == x_A, name="Region0_fA_eq_xA")

# Region 1: y_A = 1  -->  x_A >= 451,  f_A = p_A
model.addGenConstrIndicator(y_A, 1, x_A >= flow_threshold_A + 1, name="Region1_xA_ge_451")
model.addGenConstrIndicator(y_A, 1, f_A == p_A, name="Region1_fA_eq_pA")

# ==============================
# 7. Solve the model and print results
# ==============================
model.optimize()

if model.Status == GRB.OPTIMAL:
    flow_A = x_A.X
    flow_B = x_B.X
    adjusted_flow_A = f_A.X
    total_cost = model.ObjVal

    print(f"Optimal solution found.")
    print(f"Flow of A (x_A): {flow_A:.6f}")
    print(f"Flow of B (x_B): {flow_B:.6f}")
    print(f"Adjusted flow of A for cost (f_A): {adjusted_flow_A:.6f}")
    print(f"Total cost: {total_cost:.6f}")
else:
    print(f"Optimization ended with status {model.Status}")
    flow_A = float('nan')

# Final answer: flow rate of A
print(f"FinalAnswer=【{flow_A}】")