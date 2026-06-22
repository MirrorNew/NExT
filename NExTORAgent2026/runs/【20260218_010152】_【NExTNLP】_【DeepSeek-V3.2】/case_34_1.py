import gurobipy as gp

# Define all parameters
num_design_variables = 7
units = "inches"
model_coefficients = [0.7854, 3.3333, 14.9334, -43.0934, -1.508, 7.4777, 0.7854]
meshing_strength_coefficient = 27
meshing_strength_threshold = 1
shaft_strength_coefficient = 397.5
shaft_strength_threshold = 1
pitch_circle_diameter_max = 40

# Table_1_sizeRestrictions
size_restrictions = {
    'Tooth width': [2.5, 3.5],
    'Module': [0.6, 0.8],
    'Number of teeth (integer)': [17, 28],
    'Shaft 1 length': [7, 9],
    'Shaft 2 length': [7.5, 9],
    'Shaft 1 diameter': [2.5, 3.5],
    'Shaft 2 diameter': [5, 6]
}

# Create model
model = gp.Model("Reducer_Design_Optimization")
model.Params.NonConvex = 2

# Create decision variables
x1 = model.addVar(lb=2.5, ub=3.5, name="x1")
x2 = model.addVar(lb=0.6, ub=0.8, name="x2")
x3 = model.addVar(lb=17, ub=28, vtype=gp.GRB.INTEGER, name="x3")
x4 = model.addVar(lb=7, ub=9, name="x4")
x5 = model.addVar(lb=7.5, ub=9, name="x5")
x6 = model.addVar(lb=2.5, ub=3.5, name="x6")
x7 = model.addVar(lb=5, ub=6, name="x7")

# Create auxiliary variables
y1 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="y1")  # x2^2
y2 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="y2")  # x6^2
y3 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="y3")  # x7^2
y4 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="y4")  # x6^3
y5 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="y5")  # x7^3
y6 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="y6")  # x3^2
y7 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="y7")  # x1*y1*x3
y8 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="y8")  # x1*y1*y6
y9 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="y9")  # 0.7854*x1*y1*(3.3333*y6 + 14.9334*x3 - 43.0934)
y10 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="y10")  # x4*y2
y11 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="y11")  # x5*y3
y12 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="y12")  # 1.508*x1*(y2 + y3)

# Add power constraints
model.addGenConstrPow(x2, y1, 2, "x2_squared")
model.addGenConstrPow(x6, y2, 2, "x6_squared")
model.addGenConstrPow(x7, y3, 2, "x7_squared")
model.addGenConstrPow(x6, y4, 3, "x6_cubed")
model.addGenConstrPow(x7, y5, 3, "x7_cubed")
model.addGenConstrPow(x3, y6, 2, "x3_squared")

# Add product constraints
model.addConstr(y7 == x1 * y1 * x3, "gear_meshing_strength_product")
model.addConstr(y8 == x1 * y1 * y6, "shaft_strength_product")
model.addConstr(y10 == x4 * y2, "shaft1_product")
model.addConstr(y11 == x5 * y3, "shaft2_product")

# Add complex term constraints
model.addConstr(y9 == 0.7854 * x1 * y1 * (3.3333 * y6 + 14.9334 * x3 - 43.0934), "first_term")
model.addConstr(y12 == 1.508 * x1 * (y2 + y3), "second_term")

# Set up objective function
objective = y9 - y12 + 7.4777 * (y4 + y5) + 0.7854 * (y10 + y11)
model.setObjective(objective, gp.GRB.MINIMIZE)

# Add constraints
# Gear meshing strength constraint
model.addConstr(y7 >= 27, "gear_meshing_strength")

# Shaft strength constraint
model.addConstr(y8 >= 397.5, "shaft_strength")

# Pitch circle diameter constraint
model.addConstr(x2 * x3 <= 40, "pitch_circle_diameter")

# Tooth width vs module constraint
model.addConstr(x1 >= 2 * x2, "tooth_width_module")

# Solve the model
model.optimize()

# Print results
if model.status == gp.GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"Minimum weight f = {model.objVal:.4f}")
    print("Optimal design variables:")
    print(f"  x1 (tooth width) = {x1.X:.4f}")
    print(f"  x2 (module) = {x2.X:.4f}")
    print(f"  x3 (number of teeth) = {x3.X:.0f}")
    print(f"  x4 (shaft 1 length) = {x4.X:.4f}")
    print(f"  x5 (shaft 2 length) = {x5.X:.4f}")
    print(f"  x6 (shaft 1 diameter) = {x6.X:.4f}")
    print(f"  x7 (shaft 2 diameter) = {x7.X:.4f}")
    
    # Check constraints
    print("\nConstraint values:")
    print(f"  Gear meshing strength: {y7.X:.4f} (>= 27)")
    print(f"  Shaft strength: {y8.X:.4f} (>= 397.5)")
    print(f"  Pitch circle diameter: {x2.X * x3.X:.4f} (<= 40)")
    print(f"  Tooth width vs module: {x1.X:.4f} >= {2 * x2.X:.4f}")
    
    print(f"FinalAnswer=【{model.objVal:.4f}】")
else:
    print(f"No optimal solution found. Status: {model.status}")
    print(f"FinalAnswer=【No optimal solution】")