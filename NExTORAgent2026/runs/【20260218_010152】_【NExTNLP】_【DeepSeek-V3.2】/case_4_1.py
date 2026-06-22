import gurobipy as gp
import math

# Parameters from the provided list
x2_proportion_range = [0.3, 0.6]  # [lower, upper]
demand_params = {'x1': {'intercept': 5000, 'price_coef': 7}, 
                 'x2': {'intercept': 1000, 'price_coef': 10}}
equipment_hours = {'Lathe I': {'Equipment A': 3, 'Equipment B': 2, 'Equipment C': 15}, 
                   'Lathe II': {'Equipment A': 4, 'Equipment B': 1, 'Equipment C': 2}}
available_hours = {'Equipment A': 1600, 'Equipment B': 600, 'Equipment C': 750}

# Create model
model = gp.Model("ProductPortfolioOptimization")

# Set parameter for handling non-convex (bilinear) terms
model.Params.NonConvex = 2

# Decision variables
x1 = model.addVar(lb=0, ub=5000, vtype=gp.GRB.INTEGER, name="x1")
x2 = model.addVar(lb=0, ub=1000, vtype=gp.GRB.INTEGER, name="x2")
P1 = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="P1")
P2 = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="P2")

# Auxiliary variables for bilinear terms in objective
Y1 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="Y1")
Y2 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="Y2")

# Objective: maximize total sales price = P1*x1 + P2*x2 = Y1 + Y2
model.setObjective(Y1 + Y2, sense=gp.GRB.MAXIMIZE)

# Constraints

# 1. Equipment capacity constraints
model.addConstr(3*x1 + 4*x2 <= available_hours['Equipment A'], name="Equipment_A_capacity")
model.addConstr(2*x1 + x2 <= available_hours['Equipment B'], name="Equipment_B_capacity")
model.addConstr(15*x1 + 2*x2 <= available_hours['Equipment C'], name="Equipment_C_capacity")

# 2. Product II share constraints
# x2 >= 0.3*(x1 + x2)  =>  0.7*x2 - 0.3*x1 >= 0
model.addConstr(0.7*x2 - 0.3*x1 >= 0, name="Product_II_share_lower")
# x2 <= 0.6*(x1 + x2)  =>  0.4*x2 - 0.6*x1 <= 0
model.addConstr(0.4*x2 - 0.6*x1 <= 0, name="Product_II_share_upper")

# 3. Demand functions
model.addConstr(x1 + demand_params['x1']['price_coef'] * P1 == demand_params['x1']['intercept'], 
                name="Demand_function_I")
model.addConstr(x2 + demand_params['x2']['price_coef'] * P2 == demand_params['x2']['intercept'], 
                name="Demand_function_II")

# 4. Auxiliary constraints for bilinear terms Y1 = P1*x1 and Y2 = P2*x2
model.addConstr(Y1 == P1 * x1, name="Auxiliary_Y1")
model.addConstr(Y2 == P2 * x2, name="Auxiliary_Y2")

# Solve the model
model.optimize()

# Print results
if model.status == gp.GRB.OPTIMAL:
    print("Optimal solution found:")
    print(f"x1 (Lathe I quantity) = {x1.X:.0f}")
    print(f"x2 (Lathe II quantity) = {x2.X:.0f}")
    print(f"P1 (Lathe I price) = {P1.X:.2f}")
    print(f"P2 (Lathe II price) = {P2.X:.2f}")
    print(f"Total sales price = {model.ObjVal:.2f}")
    
    # Calculate total output for verification
    total_output = x1.X + x2.X
    if total_output > 0:
        x2_proportion = x2.X / total_output
        print(f"Total output = {total_output:.0f}")
        print(f"x2 proportion = {x2_proportion:.3f} (should be in [{x2_proportion_range[0]}, {x2_proportion_range[1]}])")
    
    # Equipment utilization
    print("\nEquipment utilization:")
    print(f"Equipment A: {3*x1.X + 4*x2.X:.0f} / {available_hours['Equipment A']}")
    print(f"Equipment B: {2*x1.X + x2.X:.0f} / {available_hours['Equipment B']}")
    print(f"Equipment C: {15*x1.X + 2*x2.X:.0f} / {available_hours['Equipment C']}")
    
    # The question asks to maximize total sales price, so the answer is the optimal objective value
    print(f"FinalAnswer=【{model.ObjVal:.2f}】")
else:
    print(f"No optimal solution found. Status: {model.status}")
    print(f"FinalAnswer=【No feasible solution】")