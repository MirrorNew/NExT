import gurobipy as gp

# Parameters List
budget_upper_limit = 20
A = [50, 30]
k = [0.3, 0.6]

# Create model
model = gp.Model("R&D_Investment_Optimization")

# Create decision variables
x1 = model.addVar(lb=0, ub=20, name="x1")
x2 = model.addVar(lb=0, ub=20, name="x2")

# Create auxiliary substitution variables for exponential terms
# y1 = exp(-0.3*x1), y2 = exp(-0.6*x2)
y1 = model.addVar(lb=0, ub=1, name="y1")
y2 = model.addVar(lb=0, ub=1, name="y2")

# Set non-convex parameter for nonlinear optimization
model.Params.NonConvex = 2

# Add exponential constraints using general constraints
model.addGenConstrExp(-0.3 * x1, y1, name="exp_constr1")
model.addGenConstrExp(-0.6 * x2, y2, name="exp_constr2")

# Set up objective function: max Z = 50*(1-y1) + 30*(1-y2) = 80 - 50*y1 - 30*y2
objective = 80 - 50*y1 - 30*y2
model.setObjective(objective, gp.GRB.MAXIMIZE)

# Add constraints
# Budget constraint: x1 + x2 ≤ 20
model.addConstr(x1 + x2 <= budget_upper_limit, name="budget_constraint")

# Solve the model
model.optimize()

# Print results
if model.status == gp.GRB.OPTIMAL:
    print("Optimal solution found:")
    print(f"x1 (Investment in Project 1) = {x1.X:.6f} million USD")
    print(f"x2 (Investment in Project 2) = {x2.X:.6f} million USD")
    print(f"Total investment = {x1.X + x2.X:.6f} million USD")
    print(f"Maximum total return Z = {model.ObjVal:.6f} million USD")
    
    # Output the answer to the question: the best solution (investment amounts)
    print(f"FinalAnswer=【x1={x1.X:.6f}, x2={x2.X:.6f}, Z={model.ObjVal:.6f}】")
else:
    print("No optimal solution found")
    print(f"FinalAnswer=【No optimal solution found】")