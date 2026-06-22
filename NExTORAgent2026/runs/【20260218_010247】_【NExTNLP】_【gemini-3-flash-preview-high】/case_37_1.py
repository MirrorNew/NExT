import gurobipy as gp
from gurobipy import GRB

# 2. Define all parameter matrices and data inputs
# From Parameters List
octane_threshold = 80
unit_price_A = 30
quality_index_A = 90
unit_price_B = 20
quality_index_B = 70
quality_index_min = 80
total_flow = 1000
flow_threshold_A = 450
penalty_exponent_A = 1.05

# 3. Create decision variables
model = gp.Model("Refining_Optimization")

# Decision variables for raw materials flow rate
x_A = model.addVar(lb=0, ub=1000, vtype=GRB.CONTINUOUS, name="x_A")
x_B = model.addVar(lb=0, ub=1000, vtype=GRB.CONTINUOUS, name="x_B")

# 4. Create auxiliary substitution or indicator variables
# v_pow is the term x_A^1.05
v_pow = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v_pow")
# f_A is the adjusted flow rate of A based on the threshold
f_A = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="f_A")
# y_A is the binary indicator: 1 if x_A > 450, 0 otherwise
y_A = model.addVar(vtype=GRB.BINARY, name="y_A")

# Enable solving non-convex power functions and indicator logic
model.Params.NonConvex = 2

# 5. Set up the objective function
# Minimize total raw material cost
model.setObjective(unit_price_A * f_A + unit_price_B * x_B, GRB.MINIMIZE)

# 6. Add all constraints
# Total Flow Constraint: x_A + x_B = 1000
model.addConstr(x_A + x_B == total_flow, name="TotalFlowConstraint")

# Mixture Octane Constraint: (90*x_A + 70*x_B)/(x_A + x_B) >= 80
# Simplified to: 90*x_A + 70*x_B >= 80 * (x_A + x_B)
model.addConstr(quality_index_A * x_A + quality_index_B * x_B >= quality_index_min * (x_A + x_B), name="OctaneConstraint")

# Power Calculation: v_pow = x_A ^ 1.05
model.addGenConstrPow(x_A, v_pow, penalty_exponent_A, name="PowerCalculation")

# Link binary variable y_A to flow rate x_A
# y_A=1 means x_A > 450 (using 450.0001 for strict inequality boundary in optimization)
# y_A=0 means x_A <= 450
model.addGenConstrIndicator(y_A, 1, x_A >= flow_threshold_A + 0.0001, name="Indicator_xA_GT_450")
model.addGenConstrIndicator(y_A, 0, x_A <= flow_threshold_A, name="Indicator_xA_LE_450")

# Logic for adjusted flow f_A based on indicator y_A
# If y_A = 1, then f_A = x_A^1.05 (represented by v_pow)
model.addGenConstrIndicator(y_A, 1, f_A == v_pow, name="fA_as_pow")
# If y_A = 0, then f_A = x_A
model.addGenConstrIndicator(y_A, 0, f_A == x_A, name="fA_as_linear")

# 7. Solve the model
model.optimize()

# Print results and output final answer
if model.status == GRB.OPTIMAL:
    ans_x_A = x_A.X
    print(f"FinalAnswer=【{ans_x_A}】")
else:
    print("Optimization was not successful.")