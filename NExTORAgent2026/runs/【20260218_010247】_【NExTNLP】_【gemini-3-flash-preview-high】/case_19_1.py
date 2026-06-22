import gurobipy as gp
from gurobipy import GRB

# 2. Define all parameter matrices and data inputs
# Parameters List: 
# [{'Name': 'total_water_resources', 'Type': 'integer', 'Value': 100}, 
#  {'Name': 'number_of_farms', 'Type': 'integer', 'Value': 3}, 
#  {'Name': 'a', 'Type': 'list', 'Value': [5, 3, 4]}, 
#  {'Name': 'total_irrigation_amount_limit', 'Type': 'integer', 'Value': 100}, 
#  {'Name': 'sum_of_squares_of_irrigation_water_limit', 'Type': 'integer', 'Value': 3500}]

total_water_resources = 100
number_of_farms = 3
a = [5, 3, 4]
total_irrigation_limit = 100
sum_of_squares_limit = 3500

# Create the model
model = gp.Model("IrrigationOptimization")

# Identify any function expressions that require auxiliary substitution variables, 
# and use "model.Params.NonConvex = 2" if needed.
model.Params.NonConvex = 2

# 3. Create decision variables
# w_j: irrigation water allocated to farm j (w_j >= 0)
w = model.addVars(number_of_farms, lb=0, name="w")

# 4. Create auxiliary substitution variables in coding advice
# The values of these auxiliary variables range from negative infinity to positive infinity
y = model.addVars(number_of_farms, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y") # For power terms in objective
q = model.addVars(number_of_farms, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="q") # For squared terms in constraint

# Add general power constraints for the auxiliary variables
for i in range(number_of_farms):
    # Farm index j starts from 1, so power is (i+1)/4
    power_val = (i + 1) / 4.0
    # y_i = w_i^((i+1)/4)
    model.addGenConstrPow(w[i], y[i], power_val)
    # q_i = w_i^2
    model.addGenConstrPow(w[i], q[i], 2.0)

# 5. Set up the objective function
# Maximize total yield Z = sum_{j=1}^3 a_j * w_j^(j/4)
total_yield = gp.quicksum(a[i] * y[i] for i in range(number_of_farms))
model.setObjective(total_yield, GRB.MAXIMIZE)

# 6. Add all constraints
# Constraint 1: Total water availability (sum of w_j <= 100)
model.addConstr(gp.quicksum(w[i] for i in range(number_of_farms)) <= total_irrigation_limit)

# Constraint 2: Quadratic water-usage limit (sum of w_j^2 <= 3500)
# Using auxiliary substitution variables q_j = w_j^2
model.addConstr(gp.quicksum(q[i] for i in range(number_of_farms)) <= sum_of_squares_limit)

# 7. Solve the model and print results
model.optimize()

if model.Status == GRB.OPTIMAL:
    final_yield = model.ObjVal
    print(f"FinalAnswer=【{final_yield}】")
else:
    print("FinalAnswer=【No optimal solution found】")