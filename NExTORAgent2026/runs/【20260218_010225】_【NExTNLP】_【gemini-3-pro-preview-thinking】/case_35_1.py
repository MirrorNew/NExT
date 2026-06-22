import gurobipy as gp
from gurobipy import GRB

# Create the Gurobi model
model = gp.Model("Gear_System_Optimization")

# Set NonConvex parameter to 2 to handle quadratic constraints (bilinear products)
model.Params.NonConvex = 2

# 1. Define Parameters based on the provided list
deviation_threshold = 0.005
error_tolerance = 0.001
R_exp = 0.144279
Table_1_Tooth_Range = {'x1': [12, 50], 'x2': [20, 40], 'x3': [10, 50], 'x4': [30, 60]}

# 2. Create Decision Variables
# Integer variables for the number of teeth
x1 = model.addVar(lb=Table_1_Tooth_Range['x1'][0], ub=Table_1_Tooth_Range['x1'][1], vtype=GRB.INTEGER, name="x1")
x2 = model.addVar(lb=Table_1_Tooth_Range['x2'][0], ub=Table_1_Tooth_Range['x2'][1], vtype=GRB.INTEGER, name="x2")
x3 = model.addVar(lb=Table_1_Tooth_Range['x3'][0], ub=Table_1_Tooth_Range['x3'][1], vtype=GRB.INTEGER, name="x3")
x4 = model.addVar(lb=Table_1_Tooth_Range['x4'][0], ub=Table_1_Tooth_Range['x4'][1], vtype=GRB.INTEGER, name="x4")

# 3. Create Auxiliary Substitution Variables
# x_num represents the numerator product: x2 * x3
# Bounds estimation: min=20*10=200, max=40*50=2000
x_num = model.addVar(lb=200, ub=2000, vtype=GRB.CONTINUOUS, name="x_num")

# x_den represents the denominator product: x1 * x4
# Bounds estimation: min=12*30=360, max=50*60=3000
x_den = model.addVar(lb=360, ub=3000, vtype=GRB.CONTINUOUS, name="x_den")

# R represents the actual transmission ratio
# Since R is derived from positive integers, R >= 0
R = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="R")

# diff represents the deviation (R_exp - R). Can be negative or positive.
diff = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="diff")

# f represents the squared deviation (objective)
f = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="f")

# 4. Set Objective Function
# Minimize the squared deviation
model.setObjective(f, GRB.MINIMIZE)

# 5. Add Constraints
# Constraint: x_num = x2 * x3
model.addConstr(x_num == x2 * x3, name="Constr_Numerator_Product")

# Constraint: x_den = x1 * x4
model.addConstr(x_den == x1 * x4, name="Constr_Denominator_Product")

# Constraint: R = x_num / x_den  =>  R * x_den = x_num (Avoid division by using multiplication)
model.addConstr(R * x_den == x_num, name="Constr_Ratio_Definition")

# Constraint: diff = R_exp - R
model.addConstr(diff == R_exp - R, name="Constr_Diff_Definition")

# General Constraint: f = diff^2
model.addGenConstrPow(diff, f, 2, name="GenConstr_Square_Deviation")

# 6. Solve the model
model.optimize()

# 7. Print Results
if model.status == GRB.OPTIMIZED:
    print(f"Objective Value (Squared Deviation): {model.objVal}")
    print(f"Optimal Configuration: x1={x1.X}, x2={x2.X}, x3={x3.X}, x4={x4.X}")
    print(f"Calculated R: {R.X}")
    print(f"Target R: {R_exp}")
    
    # Required output format
    print(f"FinalAnswer=【{int(round(x1.X))}】")
else:
    print("Optimization was not successful.")