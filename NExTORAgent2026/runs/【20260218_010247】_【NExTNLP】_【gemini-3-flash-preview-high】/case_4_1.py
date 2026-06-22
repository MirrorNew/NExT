import gurobipy as gp
from gurobipy import GRB

# Parameter matrices and data inputs
x2_proportion_range = [0.3, 0.6]
demand_parameters = {
    'x1': {'intercept': 5000, 'price_coef': 7},
    'x2': {'intercept': 1000, 'price_coef': 10}
}
equipment_hours = {
    'Lathe I': {'Equipment A': 3, 'Equipment B': 2, 'Equipment C': 15},
    'Lathe II': {'Equipment A': 4, 'Equipment B': 1, 'Equipment C': 2}
}
available_hours = {
    'Equipment A': 1600, 'Equipment B': 600, 'Equipment C': 750
}

# Create model
model = gp.Model("Product_Portfolio_Optimization")

# Identify any function expressions that require auxiliary substitution variables, 
# and use "model.Params.NonConvex = 2" as the model involves bilinear terms (P*x)
model.Params.NonConvex = 2

# Create decision variables
# x1 and x2 must be integers as per the problem description
x1 = model.addVar(vtype=GRB.INTEGER, lb=0, ub=5000, name="x1")
x2 = model.addVar(vtype=GRB.INTEGER, lb=0, ub=1000, name="x2")
# P1 and P2 are continuous selling prices
P1 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="P1")
P2 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="P2")

# Create auxiliary substitution variables for revenues
# R1 = P1 * x1 and R2 = P2 * x2
# Their values should range from negative infinity to positive infinity
R1 = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="R1")
R2 = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="R2")

# Set up the objective function: maximize total revenue (expected total sales price)
model.setObjective(R1 + R2, GRB.MAXIMIZE)

# Add constraints
# Equipment capacity constraints from Table 6-1
model.addConstr(
    equipment_hours['Lathe I']['Equipment A'] * x1 + 
    equipment_hours['Lathe II']['Equipment A'] * x2 <= available_hours['Equipment A'], 
    "Equipment_A_capacity"
)
model.addConstr(
    equipment_hours['Lathe I']['Equipment B'] * x1 + 
    equipment_hours['Lathe II']['Equipment B'] * x2 <= available_hours['Equipment B'], 
    "Equipment_B_capacity"
)
model.addConstr(
    equipment_hours['Lathe I']['Equipment C'] * x1 + 
    equipment_hours['Lathe II']['Equipment C'] * x2 <= available_hours['Equipment C'], 
    "Equipment_C_capacity"
)

# Product II proportion constraints (0.3 <= x2 / (x1 + x2) <= 0.6)
model.addConstr(x2 >= x2_proportion_range[0] * (x1 + x2), "Product_II_share_lower")
model.addConstr(x2 <= x2_proportion_range[1] * (x1 + x2), "Product_II_share_upper")

# Demand functions relating quantities and prices
model.addConstr(x1 == demand_parameters['x1']['intercept'] - demand_parameters['x1']['price_coef'] * P1, "Demand_function_I")
model.addConstr(x2 == demand_parameters['x2']['intercept'] - demand_parameters['x2']['price_coef'] * P2, "Demand_function_II")

# Substitution constraints for bilinear terms (P*x)
model.addConstr(R1 == P1 * x1, "Revenue_I_Substitution")
model.addConstr(R2 == P2 * x2, "Revenue_II_Substitution")

# Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print(f"Optimal Total Revenue: {model.ObjVal}")
    print(f"Lathe I quantity: {x1.X}, Price: {P1.X}")
    print(f"Lathe II quantity: {x2.X}, Price: {P2.X}")
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimization was not successful.")