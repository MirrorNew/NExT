import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
budget_upper_limit = 20
A = [50, 30]
k = [0.3, 0.6]

# 2. Initialize the model
model = gp.Model("R&D_Investment_Optimization")

# 3. Create decision variables
# x1: investment in Project 1 (anti-tumor drug), in million USD
# x2: investment in Project 2 (vaccine improvement), in million USD
x1 = model.addVar(lb=0, ub=budget_upper_limit, vtype=GRB.CONTINUOUS, name="x1")
x2 = model.addVar(lb=0, ub=budget_upper_limit, vtype=GRB.CONTINUOUS, name="x2")

# 4. Create auxiliary substitution variables
# Following advice: Introduce w1, w2 for exponents and y1, y2 for exponential terms.
# The values range from negative infinity to positive infinity as per instructions.
w1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w1")
w2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w2")
y1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y1")
y2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y2")

# 5. Set up the objective function
# Maximize Z = A[0]*(1 - exp(-k[0]*x1)) + A[1]*(1 - exp(-k[1]*x2))
# Substituting y1 = exp(w1) and y2 = exp(w2)
model.setObjective(A[0] * (1 - y1) + A[1] * (1 - y2), GRB.MAXIMIZE)

# 6. Add all constraints
# Budget constraint
model.addConstr(x1 + x2 <= budget_upper_limit, name="BudgetConstraint")

# Exponential term constraints using auxiliary variables
# w1 = -0.3 * x1
model.addConstr(w1 == -k[0] * x1, name="w1_definition")
# w2 = -0.6 * x2
model.addConstr(w2 == -k[1] * x2, name="w2_definition")

# y1 = exp(w1) and y2 = exp(w2)
model.addGenConstrExp(w1, y1, name="y1_exp")
model.addGenConstrExp(w2, y2, name="y2_exp")

# Set non-convex parameter for general functions
model.Params.NonConvex = 2

# 7. Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    total_return = model.ObjVal
    print(f"Optimal Investment in Project 1: {x1.X:.4f} million")
    print(f"Optimal Investment in Project 2: {x2.X:.4f} million")
    print(f"Total Return: {total_return:.4f} million")
    print(f"FinalAnswer=【{total_return}】")
else:
    print("Optimal solution not found.")