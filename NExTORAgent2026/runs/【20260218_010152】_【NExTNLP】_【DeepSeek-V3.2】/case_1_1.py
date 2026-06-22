import gurobipy as gp

# Define model
model = gp.Model("ProductionInventory")

# Parameters from list
b = 200
c = 100
y0 = 0
d1 = 150
d2 = 200
d3 = 212
f1_coeff = 100
f1_power = 0.9
f2_coeff = 100
f2_power = 0.8
f3_coeff = 150
f3_power = 0.5
g_coeff = 20

# Decision variables
x1 = model.addVar(lb=0, ub=b, name="x1")
x2 = model.addVar(lb=0, ub=b, name="x2")
x3 = model.addVar(lb=0, ub=b, name="x3")
y1 = model.addVar(lb=0, ub=c, name="y1")
y2 = model.addVar(lb=0, ub=c, name="y2")
y3 = model.addVar(lb=0, ub=c, name="y3")

# Auxiliary variables for power functions (must have lb=-GRB.INFINITY)
z1 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="z1")
z2 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="z2")
z3 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="z3")

# Set model to handle non-convex terms
model.Params.NonConvex = 2

# Add power constraints
model.addGenConstrPow(x1, z1, f1_power, "z1_power")
model.addGenConstrPow(x2, z2, f2_power, "z2_power")
model.addGenConstrPow(x3, z3, f3_power, "z3_power")

# Add inventory balance constraints
model.addConstr(y0 + x1 - d1 == y1, "inventory_balance_1")
model.addConstr(y1 + x2 - d2 == y2, "inventory_balance_2")
model.addConstr(y2 + x3 - d3 == y3, "inventory_balance_3")

# Add ending inventory constraint
model.addConstr(y3 == 0, "ending_inventory")

# Set objective
obj = f1_coeff * z1 + f2_coeff * z2 + f3_coeff * z3 + g_coeff * (y1 + y2 + y3)
model.setObjective(obj, gp.GRB.MINIMIZE)

# Solve model
model.optimize()

# Output results
print("Optimal solution:")
print(f"x1 = {x1.X:.4f}")
print(f"x2 = {x2.X:.4f}")
print(f"x3 = {x3.X:.4f}")
print(f"y1 = {y1.X:.4f}")
print(f"y2 = {y2.X:.4f}")
print(f"y3 = {y3.X:.4f}")
print(f"Total Cost = {model.objVal:.4f}")

# The question asks for the monthly production volumes that minimize cost
# Format as requested: three production volumes separated by commas
production_plan = f"{x1.X:.4f}, {x2.X:.4f}, {x3.X:.4f}"
print(f"FinalAnswer=【{production_plan}】")