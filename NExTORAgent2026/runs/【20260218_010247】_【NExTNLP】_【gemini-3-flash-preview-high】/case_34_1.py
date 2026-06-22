import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
# model_coefficients = [0.7854, 3.3333, 14.9334, -43.0934, -1.508, 7.4777, 0.7854]
c = [0.7854, 3.3333, 14.9334, -43.0934, -1.508, 7.4777, 0.7854]
meshing_strength_coeff = 27
shaft_strength_coeff = 397.5
pitch_circle_max = 40

# Create the Gurobi model
model = gp.Model("ReducerOptimization")
model.Params.NonConvex = 2

# 2. Create decision variables.
x1 = model.addVar(lb=2.5, ub=3.5, vtype=GRB.CONTINUOUS, name="x1")
x2 = model.addVar(lb=0.6, ub=0.8, vtype=GRB.CONTINUOUS, name="x2")
x3 = model.addVar(lb=17, ub=28, vtype=GRB.INTEGER, name="x3")
x4 = model.addVar(lb=7, ub=9, vtype=GRB.CONTINUOUS, name="x4")
x5 = model.addVar(lb=7.5, ub=9, vtype=GRB.CONTINUOUS, name="x5")
x6 = model.addVar(lb=2.5, ub=3.5, vtype=GRB.CONTINUOUS, name="x6")
x7 = model.addVar(lb=5, ub=6, vtype=GRB.CONTINUOUS, name="x7")

# 3. Create any auxiliary substitution variables (lb=-GRB.INFINITY, ub=GRB.INFINITY).
y1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y1") # x2^2
y2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y2") # x3^2
y3 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y3") # x6^2
y4 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y4") # x7^2
y5 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y5") # x6^3
y6 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y6") # x7^3
z1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="z1") # x1 * y1 = x1 * x2^2
z2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="z2") # z1 * y2 = x1 * x2^2 * x3^2
z3 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="z3") # z1 * x3 = x1 * x2^2 * x3
z5 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="z5") # x1 * y3 = x1 * x6^2
z6 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="z6") # x1 * y4 = x1 * x7^2
z7 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="z7") # x4 * y3 = x4 * x6^2
z8 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="z8") # x5 * y4 = x5 * x7^2
z9 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="z9") # x2 * x3

# Define relationships for auxiliary variables
model.addGenConstrPow(x2, y1, 2)
model.addGenConstrPow(x3, y2, 2)
model.addGenConstrPow(x6, y3, 2)
model.addGenConstrPow(x7, y4, 2)
model.addGenConstrPow(x6, y5, 3)
model.addGenConstrPow(x7, y6, 3)

model.addConstr(z1 == x1 * y1)
model.addConstr(z2 == z1 * y2)
model.addConstr(z3 == z1 * x3)
model.addConstr(z5 == x1 * y3)
model.addConstr(z6 == x1 * y4)
model.addConstr(z7 == x4 * y3)
model.addConstr(z8 == x5 * y4)
model.addConstr(z9 == x2 * x3)

# 4. Set up the objective function.
# f = 0.7854*x1*x2^2*(3.3333*x3^2 + 14.9334*x3 - 43.0934) - 1.508*x1*(x6^2+x7^2) + 7.4777*(x6^3+x7^3) + 0.7854*(x4*x6^2 + x5*x7^2)
# Using substitution and indexing from c:
# obj = c[0]*(c[1]*z2 + c[2]*z3 + c[3]*z1) + c[4]*(z5 + z6) + c[5]*(y5 + y6) + c[6]*(z7 + z8)
objective = (c[0] * (c[1] * z2 + c[2] * z3 + c[3] * z1) + 
             c[4] * (z5 + z6) + 
             c[5] * (y5 + y6) + 
             c[6] * (z7 + z8))
model.setObjective(objective, GRB.MINIMIZE)

# 5. Add all constraints.
# Gear-meshing strength limit
model.addConstr(z3 >= meshing_strength_coeff, name="gear_strength")
# Shaft-strength limit
model.addConstr(z2 >= shaft_strength_coeff, name="shaft_strength")
# Pitch circle diameter limit
model.addConstr(z9 <= pitch_circle_max, name="pitch_diameter")
# Tooth width vs module constraint
model.addConstr(x1 >= 2 * x2, name="width_vs_module")

# 6. Solve the model and print results.
model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("Optimal solution not found.")