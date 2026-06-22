import gurobipy as gp
from gurobipy import GRB

# 1. Parameter Definitions
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
    'Equipment A': 1600,
    'Equipment B': 600,
    'Equipment C': 750
}

# 2. Model Initialization
model = gp.Model("Xie_Company_Optimization")
model.Params.NonConvex = 2  # Enable non-convex constraints for quadratic auxiliary variables

# 3. Decision Variables
# Production quantities (Integer)
x1 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x1")
x2 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x2")

# 4. Auxiliary Substitution Variables for Quadratic Terms
# We need variables for x1^2 and x2^2 to linearize/handle the objective function
x1_sq = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x1_sq")
x2_sq = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x2_sq")

# 5. Constraints

# 5.1 Quadratic Constraints (GenConstr)
# Define the relationship for the squares: x_sq = x^2
model.addGenConstrPow(x1, x1_sq, 2, "sq_x1")
model.addGenConstrPow(x2, x2_sq, 2, "sq_x2")

# 5.2 Capacity Constraints
# Equipment A
model.addConstr(
    equipment_hours['Lathe I']['Equipment A'] * x1 + 
    equipment_hours['Lathe II']['Equipment A'] * x2 <= available_hours['Equipment A'],
    "Capacity_Equipment_A"
)

# Equipment B
model.addConstr(
    equipment_hours['Lathe I']['Equipment B'] * x1 + 
    equipment_hours['Lathe II']['Equipment B'] * x2 <= available_hours['Equipment B'],
    "Capacity_Equipment_B"
)

# Equipment C
model.addConstr(
    equipment_hours['Lathe I']['Equipment C'] * x1 + 
    equipment_hours['Lathe II']['Equipment C'] * x2 <= available_hours['Equipment C'],
    "Capacity_Equipment_C"
)

# 5.3 Proportion Constraints for Product II (Lathe II)
# x2 >= 0.3 * (x1 + x2)
model.addConstr(x2 >= x2_proportion_range[0] * (x1 + x2), "Proportion_Lower_Bound")

# x2 <= 0.6 * (x1 + x2)
model.addConstr(x2 <= x2_proportion_range[1] * (x1 + x2), "Proportion_Upper_Bound")

# 6. Objective Function
# Maximize Total Revenue Z = P1*x1 + P2*x2
# From demand functions: 
# x1 = 5000 - 7*P1 => P1 = (5000 - x1) / 7
# x2 = 1000 - 10*P2 => P2 = (1000 - x2) / 10
# Revenue1 = ((5000 - x1) / 7) * x1 = (5000*x1 - x1^2) / 7
# Revenue2 = ((1000 - x2) / 10) * x2 = (1000*x2 - x2^2) / 10

# Using auxiliary variables x1_sq and x2_sq for x1^2 and x2^2
revenue_x1 = (demand_parameters['x1']['intercept'] * x1 - x1_sq) / demand_parameters['x1']['price_coef']
revenue_x2 = (demand_parameters['x2']['intercept'] * x2 - x2_sq) / demand_parameters['x2']['price_coef']

model.setObjective(revenue_x1 + revenue_x2, GRB.MAXIMIZE)

# 7. Solve and Output
model.optimize()

if model.Status == GRB.OPTIMAL:
    # Calculate derived values for reporting (optional context, but good for verification)
    optimized_x1 = x1.X
    optimized_x2 = x2.X
    p1 = (demand_parameters['x1']['intercept'] - optimized_x1) / demand_parameters['x1']['price_coef']
    p2 = (demand_parameters['x2']['intercept'] - optimized_x2) / demand_parameters['x2']['price_coef']
    
    print(f"Optimal Solution Found:")
    print(f"Quantity Lathe I (x1): {optimized_x1}")
    print(f"Quantity Lathe II (x2): {optimized_x2}")
    print(f"Price Lathe I (P1): {p1:.2f}")
    print(f"Price Lathe II (P2): {p2:.2f}")
    print(f"Total Revenue: {model.ObjVal}")
    
    # Required output format
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("No optimal solution found.")