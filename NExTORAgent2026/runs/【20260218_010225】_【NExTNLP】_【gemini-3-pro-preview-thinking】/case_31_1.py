import gurobipy as gp
from gurobipy import GRB

# 1. Initialize Model
model = gp.Model("Kazdale_Farm_Optimization")

# 2. Parameters
total_fertilizer_capacity = 150
max_fertilizer_field1 = 100
max_fertilizer_field2 = 90
yield_coeff_field1 = 5
yield_coeff_field2 = 6
reduction_coef_field2 = 2e-05

# 3. Decision Variables
# Fertilizer input for Field 1
x1 = model.addVar(lb=0, ub=max_fertilizer_field1, vtype=GRB.CONTINUOUS, name="x1")
# Fertilizer input for Field 2
x2 = model.addVar(lb=0, ub=max_fertilizer_field2, vtype=GRB.CONTINUOUS, name="x2")

# Yield variables (y1, y2) and Reduction variable (delta2)
y1 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y1")
y2 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y2")
delta2 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="delta2")

# 4. Auxiliary Variables for Nonlinear Terms
# Auxiliary variable for sqrt(x1)
x1_sqrt = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x1_sqrt")
# Auxiliary variable for sqrt(x2)
x2_sqrt = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x2_sqrt")
# Auxiliary variable for x2^2
x2_sq = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x2_sq")

# Set NonConvex parameter to handle general constraints involving powers
model.Params.NonConvex = 2

# 5. Objective Function
# Maximize Total Net Yield: y1 + y2 - delta2
model.setObjective(y1 + y2 - delta2, GRB.MAXIMIZE)

# 6. Constraints

# Constraint 1: Total Fertilizer Allocation
model.addConstr(x1 + x2 <= total_fertilizer_capacity, "Total_Fertilizer_Capacity")

# Constraint 2 & 3: Field 1 Yield Function y1 = 5 * sqrt(x1)
# Step 2a: Define relation x1_sqrt = x1^0.5
model.addGenConstrPow(x1, x1_sqrt, 0.5, "GenConstr_Sqrt_x1")
# Step 2b: y1 = 5 * x1_sqrt
model.addConstr(y1 == yield_coeff_field1 * x1_sqrt, "Yield_Field1")

# Constraint 4 & 5: Field 2 Yield Function y2 = 6 * sqrt(x2)
# Step 3a: Define relation x2_sqrt = x2^0.5
model.addGenConstrPow(x2, x2_sqrt, 0.5, "GenConstr_Sqrt_x2")
# Step 3b: y2 = 6 * x2_sqrt
model.addConstr(y2 == yield_coeff_field2 * x2_sqrt, "Yield_Field2")

# Constraint 6: Yield Reduction Field 2 delta2 = 0.00002 * x2^2
# Step 4a: Define relation x2_sq = x2^2
model.addGenConstrPow(x2, x2_sq, 2.0, "GenConstr_Sq_x2")
# Step 4b: delta2 = 0.00002 * x2_sq
model.addConstr(delta2 == reduction_coef_field2 * x2_sq, "Reduction_Field2")

# 7. Solve the model
model.optimize()

# 8. Output Results
if model.status == GRB.OPTIMAL:
    print(f"Optimal Fertilizer Field 1 (x1): {x1.X} kg")
    print(f"Optimal Fertilizer Field 2 (x2): {x2.X} kg")
    print(f"Yield Field 1 (y1): {y1.X} tons")
    print(f"Yield Field 2 (y2): {y2.X} tons")
    print(f"Yield Reduction Field 2 (delta2): {delta2.X} tons")
    # The question answer is the objective value (Total Yield)
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimization was not successful.")