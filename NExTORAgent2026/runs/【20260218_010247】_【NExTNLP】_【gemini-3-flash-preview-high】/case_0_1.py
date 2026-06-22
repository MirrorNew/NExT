import gurobipy as gp
from gurobipy import GRB

# Parameters from the provided list
number_of_sections = 4
number_of_work_sites = 4
Q = [10, 15, 20, 25]
coordinates = [[1, 1], [1, 2], [2, 1], [2, 2]]

# Create the model
model = gp.Model("Concrete_Mixing_Station_Location")

# Decision variables for the location of the central mixing station (x, y)
x = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
y = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y")

# Auxiliary variables for distance calculation
# All auxiliary variables are set to range from negative infinity to positive infinity as per instructions
dx = model.addVars(number_of_work_sites, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="dx")
dy = model.addVars(number_of_work_sites, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="dy")
dx2 = model.addVars(number_of_work_sites, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="dx2")
dy2 = model.addVars(number_of_work_sites, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="dy2")
d_sq = model.addVars(number_of_work_sites, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="d_sq")
d = model.addVars(number_of_work_sites, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="d")

# Objective function: minimize the total transportation volume Σ_{i=1}^4 Q_i * d_i
model.setObjective(gp.quicksum(Q[i] * d[i] for i in range(number_of_work_sites)), GRB.MINIMIZE)

# Constraints for the distance definition: d_i = sqrt((x - x_i)^2 + (y - y_i)^2)
for i in range(number_of_work_sites):
    # dx_i = x - x_i
    model.addConstr(dx[i] == x - coordinates[i][0], name=f"dx_def_{i}")
    # dy_i = y - y_i
    model.addConstr(dy[i] == y - coordinates[i][1], name=f"dy_def_{i}")
    
    # dx2_i = dx_i^2
    # Following the order (X, Y, 2) for Y = X^2
    model.addGenConstrPow(dx[i], dx2[i], 2, name=f"dx2_pow_{i}")
    
    # dy2_i = dy_i^2
    model.addGenConstrPow(dy[i], dy2[i], 2, name=f"dy2_pow_{i}")
    
    # d_sq_i = dx2_i + dy2_i
    model.addConstr(d_sq[i] == dx2[i] + dy2[i], name=f"d_sq_def_{i}")
    
    # d_i = d_sq_i^0.5
    # Following the order (X, Y, 0.5) for Y = sqrt(X)
    model.addGenConstrPow(d_sq[i], d[i], 0.5, name=f"d_sqrt_{i}")

# Set non-convex parameter to 2 as the problem involves nonlinear general constraints
model.Params.NonConvex = 2

# Solve the model
model.optimize()

# Print results and final answer
if model.status == GRB.OPTIMAL:
    print(f"Optimal Location of Mixing Station: (x, y) = ({x.X:.4f}, {y.X:.4f})")
    print(f"Minimum Total Transportation Volume: {model.ObjVal:.4f}")
    # FinalAnswer output is the objective function value (total minimum transportation volume)
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimal solution not found.")