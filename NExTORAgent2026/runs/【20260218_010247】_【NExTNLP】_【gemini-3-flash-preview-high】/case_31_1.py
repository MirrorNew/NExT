import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and any other necessary packages.
# Already imported gp and GRB.

# 2. Define all parameter matrices and data inputs.
threshold_salinity_field2 = 80
total_fertilizer_capacity = 150
max_fertilizer_field1 = 100
max_fertilizer_field2 = 90
yield_coeff_field1 = 5
yield_coeff_field2 = 6
reduction_coef_field2 = 0.00002

# 3. Create model
model = gp.Model("IrrigationOptimization")
# Identify any function expressions that require auxiliary substitution variables, and use "model.Params.NonConvex = 2" if needed.
model.Params.NonConvex = 2

# Create decision variables.
x1 = model.addVar(lb=0, ub=max_fertilizer_field1, vtype=GRB.CONTINUOUS, name="x1")
x2 = model.addVar(lb=0, ub=max_fertilizer_field2, vtype=GRB.CONTINUOUS, name="x2")
y1 = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y1")
y2 = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y2")
delta2 = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="delta2")

# 4. Create any auxiliary substitution or indicator variables in coding advice
# (The values of these auxiliary variables should range from negative infinity to positive infinity, lb=-GRB.INFINITY, ub=GRB.INFINITY).
s1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="s1")
s2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="s2")
q2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="q2")

# 5. Set up the objective function.
# Farmers hope to maximize the total yield y1 + y2 by reasonably allocating fertilizers x1, x2.
model.setObjective(y1 + y2, GRB.MAXIMIZE)

# 6. Add all constraints (including gen-constr and indicator constraints).
# Total Fertilizer Allocation
model.addConstr(x1 + x2 <= total_fertilizer_capacity, name="Total_Fertilizer_Allocation")

# Auxiliary relationships for power functions
# Use model.addGenConstrPow(X, Y, a) to define the relationship Y = X^a.
model.addGenConstrPow(x1, s1, 0.5, name="sqrt_x1")
model.addGenConstrPow(x2, s2, 0.5, name="sqrt_x2")
model.addGenConstrPow(x2, q2, 2.0, name="square_x2")

# Yield Function Field 1: y1 = 5 * sqrt(x1)
model.addConstr(y1 == yield_coeff_field1 * s1, name="Yield_Function_Field1")

# Yield Reduction Field 2: delta2 = 0.00002 * x2^2
model.addConstr(delta2 == reduction_coef_field2 * q2, name="Yield_Reduction_Field2")

# Yield Function Field 2: y2 = 6 * sqrt(x2) - delta2
model.addConstr(y2 == yield_coeff_field2 * s2 - delta2, name="Net_Yield_Function_Field2")

# 7. Solve the model and print results.
model.optimize()

if model.status == GRB.OPTIMAL:
    total_yield = model.ObjVal
    print(f"FinalAnswer=【{total_yield}】")