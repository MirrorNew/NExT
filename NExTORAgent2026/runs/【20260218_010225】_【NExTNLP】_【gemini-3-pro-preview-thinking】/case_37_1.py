import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and any other necessary packages.
# Initialize the model
model = gp.Model("RefineryOptimization")

# Set NonConvex parameter to 2 to handle general nonlinear constraints (Power function)
model.Params.NonConvex = 2

# 2. Define all parameter matrices and data inputs.
# Parameters from the provided list
octane_threshold = 80
unit_price_A = 30
quality_index_A = 90
unit_price_B = 20
quality_index_B = 70
quality_index_min = 80
total_flow = 1000
flow_threshold_A = 450
penalty_exponent_A = 1.05

# 3. Create decision variables.
# x_A: Flow rate of raw material A
x_A = model.addVar(lb=0, ub=total_flow, vtype=GRB.CONTINUOUS, name="x_A")
# x_B: Flow rate of raw material B
x_B = model.addVar(lb=0, ub=total_flow, vtype=GRB.CONTINUOUS, name="x_B")
# f_A: Adjusted flow of A for cost calculation
f_A = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="f_A")

# 4. Create any auxiliary substitution or indicator variables.
# x_pow will store the value of x_A^1.05
x_pow = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x_pow")
# y is a binary indicator variable: y=1 if x_A > 450 (High Tier), y=0 if x_A <= 450 (Low Tier)
y = model.addVar(vtype=GRB.BINARY, name="y")

# 5. Set up the objective function.
# Minimize total raw-material cost = 30*f_A + 20*x_B
model.setObjective(unit_price_A * f_A + unit_price_B * x_B, GRB.MINIMIZE)

# 6. Add all constraints (including gen‐constr and indicator constraints).

# Total Flow Constraint: x_A + x_B = 1000
model.addConstr(x_A + x_B == total_flow, name="TotalFlow")

# Mixture Octane Constraint: (90*x_A + 70*x_B)/(x_A + x_B) >= 80
# Linearized: 90*x_A + 70*x_B >= 80*(x_A + x_B)
model.addConstr(quality_index_A * x_A + quality_index_B * x_B >= quality_index_min * (x_A + x_B), name="OctaneConstraint")

# Power Constraint: x_pow = x_A ^ 1.05
# Note: order is (x, y, a) for y = x^a
model.addGenConstrPow(x_A, x_pow, penalty_exponent_A, name="PowerCalc")

# Indicator Constraints for f_A logic
# If y = 0, x_A <= 450 and f_A = x_A
model.addGenConstrIndicator(y, 0, x_A <= flow_threshold_A)
model.addGenConstrIndicator(y, 0, f_A == x_A)

# If y = 1, x_A >= 450 (representing > 450 in logic) and f_A = x_pow
# Gurobi uses >= for indicator inequalities
model.addGenConstrIndicator(y, 1, x_A >= flow_threshold_A)
model.addGenConstrIndicator(y, 1, f_A == x_pow)

# 7. Solve the model and print results.
model.optimize()

if model.status == GRB.OPTIMAL:
    # Print the flow rate of A as required
    print(f"Optimal Flow Rate A: {x_A.X}")
    print(f"Optimal Flow Rate B: {x_B.X}")
    print(f"Total Cost: {model.ObjVal}")
    print(f"FinalAnswer=【{x_A.X}】")
else:
    print("No optimal solution found.")