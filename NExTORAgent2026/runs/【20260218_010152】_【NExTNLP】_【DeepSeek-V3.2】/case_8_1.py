import gurobipy as gp

# Define parameters from the provided list
price_A_coeff = [70.0, -4.0]  # [constant, coefficient for x₁]
price_B_coeff = [150.0, -15.0]  # [constant, coefficient for x₂]
unit_production_cost = 150000  # in yuan
machine_loss_fee = 1000000  # in yuan

# Note: The objective function in the mathematical model uses "10,000 yuan" as unit for price,
# and unit production cost and machine loss fee are given in yuan.
# Therefore, we need to convert costs to the same unit (10,000 yuan) for consistency.
unit_cost_10k = unit_production_cost / 10000  # 15
machine_fee_10k = machine_loss_fee / 10000  # 100

# Create model
model = gp.Model("ProfitMaximization")

# Decision variables
x1 = model.addVar(lb=0.0, name="x1")  # demand from Customer A
x2 = model.addVar(lb=0.0, name="x2")  # demand from Customer B

# Auxiliary variables for quadratic terms (as per math_model_advice)
Y1 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="Y1")  # Y1 = x1^2
Y2 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="Y2")  # Y2 = x2^2

# Since we have quadratic terms, set non-convex parameter
model.Params.NonConvex = 2

# Add constraints for quadratic terms
model.addGenConstrPow(x1, Y1, 2, "pow_constr1")
model.addGenConstrPow(x2, Y2, 2, "pow_constr2")

# Objective function: π = 70x₁ - 4x₁² + 150x₂ - 15x₂² - 15(x₁ + x₂) - 100
# Rewritten using auxiliary variables: π = 70x₁ - 4Y1 + 150x₂ - 15Y2 - 15x₁ - 15x₂ - 100
# Simplify: π = (70-15)x₁ - 4Y1 + (150-15)x₂ - 15Y2 - 100
objective = (price_A_coeff[0] - unit_cost_10k) * x1 + price_A_coeff[1] * Y1 + (price_B_coeff[0] - unit_cost_10k) * x2 + price_B_coeff[1] * Y2 - machine_fee_10k

model.setObjective(objective, gp.GRB.MAXIMIZE)

# Solve the model
model.optimize()

# Print results
if model.status == gp.GRB.OPTIMAL:
    print(f"Optimal Solution Found")
    print(f"x1 (Demand from Customer A): {x1.X:.4f}")
    print(f"x2 (Demand from Customer B): {x2.X:.4f}")
    print(f"Maximum Profit (in 10,000 yuan): {model.objVal:.4f}")
    # Output the answer as required
    print(f"FinalAnswer=【{model.objVal:.4f}】")
else:
    print(f"Model did not solve to optimality. Status: {model.status}")
    print(f"FinalAnswer=【None】")