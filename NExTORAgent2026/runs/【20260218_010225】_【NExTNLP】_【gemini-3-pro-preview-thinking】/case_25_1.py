import gurobipy as gp
from gurobipy import GRB

# 1. Initialize the Model
model = gp.Model("RD_Investment_Optimization")

# 2. Define Parameters and Data Inputs
budget_upper_limit = 20
A = [50, 30]
k = [0.3, 0.6]

# 3. Create Decision Variables
# Investments in Project 1 and Project 2
x1 = model.addVar(lb=0, ub=budget_upper_limit, vtype=GRB.CONTINUOUS, name="x1")
x2 = model.addVar(lb=0, ub=budget_upper_limit, vtype=GRB.CONTINUOUS, name="x2")

# 4. Create Auxiliary Substitution Variables
# u1, u2 represent the exponents: -k*x
# y1, y2 represent the exponential terms: exp(u)
# We set wide bounds as per instructions for auxiliary variables
u1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="u1")
y1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y1")
u2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="u2")
y2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y2")

# 5. Set Up the Objective Function
# Maximize Z = A[0]*(1 - y1) + A[1]*(1 - y2)
# Note: y1 = exp(-0.3*x1), y2 = exp(-0.6*x2)
model.setObjective(A[0] * (1 - y1) + A[1] * (1 - y2), GRB.MAXIMIZE)

# 6. Add Constraints
# Budget Constraint
model.addConstr(x1 + x2 <= budget_upper_limit, name="Budget_Constraint")

# Auxiliary constraints for Project 1
# u1 = -0.3 * x1
model.addConstr(u1 == -k[0] * x1, name="Aux_u1")
# y1 = exp(u1)
model.addGenConstrExp(u1, y1, name="Exp_y1")

# Auxiliary constraints for Project 2
# u2 = -0.6 * x2
model.addConstr(u2 == -k[1] * x2, name="Aux_u2")
# y2 = exp(u2)
model.addGenConstrExp(u2, y2, name="Exp_y2")

# Set NonConvex parameter to handle general nonlinear constraints
model.Params.NonConvex = 2

# 7. Solve the Model and Print Results
model.optimize()

if model.Status == GRB.OPTIMAL:
    print(f"Optimal Objective Value: {model.ObjVal}")
    print(f"Investment in Project 1 (x1): {x1.X}")
    print(f"Investment in Project 2 (x2): {x2.X}")
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimization was stopped with status", model.Status)