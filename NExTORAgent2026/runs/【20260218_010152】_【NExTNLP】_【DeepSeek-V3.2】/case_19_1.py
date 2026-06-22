import gurobipy as gp
from gurobipy import GRB

# Parameters
total_water_resources = 100
number_of_farms = 3
a = [5, 3, 4]
total_irrigation_amount_limit = 100
sum_of_squares_of_irrigation_water_limit = 3500

# Create model
model = gp.Model("IrrigationAllocation")

# Create decision variables
w = model.addVars(number_of_farms, lb=0, name="w")

# Create auxiliary variables for objective function terms
Y = model.addVars(number_of_farms, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Y")

# Create auxiliary variables for squared terms in constraint
Q = model.addVars(number_of_farms, lb=0, name="Q")

# Set model parameter for non-convex optimization
model.Params.NonConvex = 2

# Add power constraints for objective function terms
# For w1^(1/4)
model.addGenConstrPow(w[0], Y[0], 0.25, "pow_w1_1_4")
# For w2^(1/2)
model.addGenConstrPow(w[1], Y[1], 0.5, "pow_w2_1_2")
# For w3^(3/4)
model.addGenConstrPow(w[2], Y[2], 0.75, "pow_w3_3_4")

# Add power constraints for squared terms
model.addGenConstrPow(w[0], Q[0], 2, "pow_w1_2")
model.addGenConstrPow(w[1], Q[1], 2, "pow_w2_2")
model.addGenConstrPow(w[2], Q[2], 2, "pow_w3_2")

# Set objective function
model.setObjective(gp.quicksum(a[j] * Y[j] for j in range(number_of_farms)), GRB.MAXIMIZE)

# Add constraints
# Total water availability constraint
model.addConstr(gp.quicksum(w[j] for j in range(number_of_farms)) <= total_irrigation_amount_limit, "total_water")

# Sum of squares constraint
model.addConstr(gp.quicksum(Q[j] for j in range(number_of_farms)) <= sum_of_squares_of_irrigation_water_limit, "sum_of_squares")

# Solve the model
model.optimize()

# Check if solution is found
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"Objective value: {model.ObjVal}")
    
    total_yield = 0
    for j in range(number_of_farms):
        print(f"Farm {j+1}:")
        print(f"  Water allocation w_{j+1}: {w[j].X:.4f}")
        print(f"  Contribution to yield: {a[j] * Y[j].X:.4f}")
        total_yield += a[j] * Y[j].X
    
    print(f"Total water allocated: {sum(w[j].X for j in range(number_of_farms)):.4f}")
    print(f"Sum of squares of water allocations: {sum(Q[j].X for j in range(number_of_farms)):.4f}")
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print(f"No optimal solution found. Status: {model.status}")
    print(f"FinalAnswer=【None】")