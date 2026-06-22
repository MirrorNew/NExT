import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and any other necessary packages.
# (Already done)

# 2. Define all parameter matrices and data inputs.
E = [100, 150, 80]
alpha = [0.5, 0.8, 1.0]
R_min = 120
bonus_rate = 60

# 3. Create the model
model = gp.Model("XinyuanHongdaXinkeOptimization")

# 4. Create decision variables.
# Emission reduction of factory i (tons)
x = model.addVars(3, lb=0, ub=E, name="x")
# Total bonus awarded (10,000 yuan)
B = model.addVar(lb=0, name="B")

# Create any auxiliary substitution variables in coding advice.
# (The values of these auxiliary variables should range from negative infinity to positive infinity, lb=-GRB.INFINITY, ub=GRB.INFINITY).
# Create three continuous auxiliary variables y1, y2, y3 to represent the squared terms x1^2, x2^2, and x3^2 respectively.
y = model.addVars(3, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y")

# 5. Set up the objective function.
# Minimize Z = sum(alpha_i * x_i^2) - B
model.setObjective(gp.quicksum(alpha[i] * y[i] for i in range(3)) - B, GRB.MINIMIZE)

# 6. Add all constraints.
# Mandatory Emission Reduction Requirement: sum(x_i) >= 120
model.addConstr(gp.quicksum(x[i] for i in range(3)) >= R_min, name="TotalReduction")

# Bonus definition based on exceeding the 120 tons target
# B = 60 * (sum(x_i) - 120)
# Per context: B >= 60 * (sum(x_i) - 120) and B <= 60 * (sum(x_i) - 120)
model.addConstr(B == bonus_rate * (gp.quicksum(x[i] for i in range(3)) - R_min), name="BonusDefinition")

# Auxiliary constraints for squaring terms: y_i = x_i^2
# Gurobi's addGenConstrPow(X, Y, a) is Y = X^a
for i in range(3):
    model.addGenConstrPow(x[i], y[i], 2)

# Solve the model
model.Params.NonConvex = 2
model.optimize()

# 7. Print results and the final answer.
if model.status == GRB.OPTIMAL:
    # FinalAnswer is the minimized total cost
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found.")