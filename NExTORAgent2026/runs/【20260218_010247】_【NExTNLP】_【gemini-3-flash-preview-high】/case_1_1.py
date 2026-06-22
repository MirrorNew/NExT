import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs
b = 200
c = 100
y0_val = 0
y3_val = 0
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

# 2. Create the model
model = gp.Model("HuaxinElectronicsOptimization")

# 3. Create decision variables
x1 = model.addVar(lb=0, ub=b, vtype=GRB.CONTINUOUS, name="x1")
x2 = model.addVar(lb=0, ub=b, vtype=GRB.CONTINUOUS, name="x2")
x3 = model.addVar(lb=0, ub=b, vtype=GRB.CONTINUOUS, name="x3")
y1 = model.addVar(lb=0, ub=c, vtype=GRB.CONTINUOUS, name="y1")
y2 = model.addVar(lb=0, ub=c, vtype=GRB.CONTINUOUS, name="y2")
y3 = model.addVar(lb=0, ub=c, vtype=GRB.CONTINUOUS, name="y3")

# 4. Create auxiliary substitution variables
v1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v1")
v2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v2")
v3 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v3")

# 5. Set up the objective function
# Minimize total cost: f1(x1) + f2(x2) + f3(x3) + g(y1) + g(y2) + g(y3)
model.setObjective(f1_coeff * v1 + f2_coeff * v2 + f3_coeff * v3 + g_coeff * (y1 + y2 + y3), GRB.MINIMIZE)

# 6. Add all constraints
# Inventory Balance Constraints
model.addConstr(y0_val + x1 - d1 == y1, name="InventoryBalance1")
model.addConstr(y1 + x2 - d2 == y2, name="InventoryBalance2")
model.addConstr(y2 + x3 - d3 == y3, name="InventoryBalance3")

# Ending Inventory Constraint
model.addConstr(y3 == y3_val, name="EndingInventoryConstraint")

# General constraints for power functions (v = x^p)
model.addGenConstrPow(x1, v1, f1_power, name="ProductionCostPow1")
model.addGenConstrPow(x2, v2, f2_power, name="ProductionCostPow2")
model.addGenConstrPow(x3, v3, f3_power, name="ProductionCostPow3")

# Set the parameter to solve non-convex problems as needed for the power constraints
model.Params.NonConvex = 2

# 7. Solve the model and print results
model.optimize()

if model.Status == GRB.OPTIMAL:
    objective_value = model.ObjVal
    print(f"FinalAnswer=【{objective_value}】")
else:
    print("Optimization was not successful.")