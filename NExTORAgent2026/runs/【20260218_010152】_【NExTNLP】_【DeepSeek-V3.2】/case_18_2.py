import gurobipy as gp
from gurobipy import GRB

# Step 2: Define all parameter matrices and data inputs
E = [100, 150, 80]          # Current emissions (tons)
alpha = [0.5, 0.8, 1.0]    # Cost coefficients (10,000 yuan/ton²)
R_min = 120                 # Minimum total reduction requirement (tons)
bonus_rate = 60            # Bonus rate (10,000 yuan per ton excess)

# Step 3: Create decision variables
model = gp.Model("EmissionReduction")

# Emission reduction variables
x1 = model.addVar(lb=0, ub=E[0], name="x1")
x2 = model.addVar(lb=0, ub=E[1], name="x2")
x3 = model.addVar(lb=0, ub=E[2], name="x3")

# Bonus variable
B = model.addVar(lb=0, ub=GRB.INFINITY, name="B")

# Step 4: Create auxiliary substitution variables
# Auxiliary variables for x_i^2 terms
y1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y1")
y2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y2")
y3 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y3")

# Auxiliary variable for total reduction S = x1 + x2 + x3
S = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="S")

# Auxiliary variable for excess reduction E_excess = max(S - 120, 0)
E_excess = model.addVar(lb=0, ub=GRB.INFINITY, name="E_excess")

# Step 5: Set up the objective function
model.setObjective(alpha[0]*y1 + alpha[1]*y2 + alpha[2]*y3 - B, GRB.MINIMIZE)

# Step 6: Add all constraints
# Enable non-convex optimization for quadratic terms
model.Params.NonConvex = 2

# Add quadratic constraints: y_i = x_i^2
model.addGenConstrPow(x1, y1, 2, "quad1")
model.addGenConstrPow(x2, y2, 2, "quad2")
model.addGenConstrPow(x3, y3, 2, "quad3")

# Add constraint for total reduction S
model.addConstr(S == x1 + x2 + x3, "total_reduction")

# Add minimum total reduction constraint
model.addConstr(S >= R_min, "min_reduction")

# Add constraints for excess reduction E_excess = max(S - 120, 0)
# This is equivalent to: E_excess >= S - R_min and E_excess >= 0
# and minimizing B (which is bonus_rate * E_excess) will force E_excess to be as small as possible
model.addConstr(E_excess >= S - R_min, "excess_lower")
model.addConstr(E_excess >= 0, "excess_nonneg")

# Add constraint linking excess reduction to bonus
model.addConstr(B == bonus_rate * E_excess, "bonus_calc")

# Step 7: Solve the model and print results
model.optimize()

# Check optimization status
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    print("\nEmission Reduction Plan:")
    print(f"Factory 1 reduction: {x1.X:.2f} tons")
    print(f"Factory 2 reduction: {x2.X:.2f} tons")
    print(f"Factory 3 reduction: {x3.X:.2f} tons")
    print(f"Total reduction: {S.X:.2f} tons")
    print(f"Excess reduction: {E_excess.X:.2f} tons")
    print(f"Total bonus: {B.X:.2f} (10,000 yuan)")
    print(f"Total cost (before bonus): {alpha[0]*y1.X + alpha[1]*y2.X + alpha[2]*y3.X:.2f} (10,000 yuan)")
    print(f"Net cost (after bonus): {model.ObjVal:.2f} (10,000 yuan)")
    
    # Format the answer as requested
    answer_str = f"x1={x1.X:.2f}, x2={x2.X:.2f}, x3={x3.X:.2f}, B={B.X:.2f}, NetCost={model.ObjVal:.2f}"
    print(f"FinalAnswer=【{answer_str}】")
else:
    print(f"Optimization failed. Status: {model.status}")