import gurobipy as gp

# Parameters from the Parameters List
unit_price_A = 30
quality_index_A = 90
unit_price_B = 20
quality_index_B = 70
quality_index_min = 80
total_flow = 1000
flow_threshold_A = 450
penalty_exponent_A = 1.05

# Create model
model = gp.Model("PetroleumRefining")

# Decision variables
x_A = model.addVar(lb=0, ub=1000, name="x_A")
x_B = model.addVar(lb=0, ub=1000, name="x_B")
f_A = model.addVar(lb=0, ub=gp.GRB.INFINITY, name="f_A")

# Auxiliary variables for power function
z_A = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="z_A")

# Binary indicator variable for x_A > 450
y = model.addVar(vtype=gp.GRB.BINARY, name="y")

# Set parameters for non-convex optimization
model.Params.NonConvex = 2

# Objective function: minimize total cost
model.setObjective(unit_price_A * f_A + unit_price_B * x_B, gp.GRB.MINIMIZE)

# Constraints

# 1. Mixture octane constraint: (90*x_A + 70*x_B)/(x_A + x_B) >= 80
# Simplifies to: 90*x_A + 70*x_B >= 80*(x_A + x_B)
# Which simplifies further to: 10*x_A - 10*x_B >= 0 => x_A >= x_B
model.addConstr(quality_index_A * x_A + quality_index_B * x_B >= 
                quality_index_min * (x_A + x_B), "octane_constraint")

# 2. Total flow constraint: x_A + x_B = 1000
model.addConstr(x_A + x_B == total_flow, "total_flow")

# 3. Power constraint: z_A = x_A^1.05
model.addGenConstrPow(x_A, z_A, penalty_exponent_A, "power_constraint")

# 4. Indicator constraints for threshold
# If y = 1, then x_A >= 450.001 (using a small epsilon to ensure strict >)
model.addGenConstrIndicator(y, 1, x_A >= 450.001, "indicator_geq")
# If y = 0, then x_A <= 450
model.addGenConstrIndicator(y, 0, x_A <= 450, "indicator_leq")

# 5. Link f_A to x_A and z_A based on indicator
M = 1000  # big-M value

# CORRECTED: f_A should equal x_A when y = 0 (x_A <= 450)
model.addConstr(f_A >= x_A - M * (1 - y), "link_fA_xA_lb")
model.addConstr(f_A <= x_A + M * (1 - y), "link_fA_xA_ub")

# CORRECTED: f_A should equal z_A when y = 1 (x_A > 450)
model.addConstr(f_A >= z_A - M * y, "link_fA_zA_lb")
model.addConstr(f_A <= z_A + M * y, "link_fA_zA_ub")

# Solve the model
model.optimize()

# Print results
if model.status == gp.GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"x_A (flow rate of raw material A) = {x_A.X}")
    print(f"x_B (flow rate of raw material B) = {x_B.X}")
    print(f"f_A (adjusted flow of A) = {f_A.X}")
    print(f"Total cost = {model.ObjVal}")
    print(f"Indicator y (1 if x_A > 450) = {y.X}")
    
    # Output the final answer as requested
    print(f"FinalAnswer=【{x_A.X}】")
else:
    print(f"Optimization failed. Status: {model.status}")