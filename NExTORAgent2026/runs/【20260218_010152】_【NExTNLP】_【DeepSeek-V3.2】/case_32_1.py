import gurobipy as gp

# Create the model
model = gp.Model("MarketingBudgetAllocation")

# Enable non-convex optimization
model.Params.NonConvex = 2

# Parameters from the list
total_budget = 100
max_A = 60
max_B = 50
max_C = 50
coeff_A = 30.23
coeff_B = 24.36
coeff_C = 20.12
expansion_factor = 3
M = 100  # Big-M constant for indicator constraints

# Decision variables
x_A = model.addVar(lb=0, ub=max_A, name="x_A")
x_B = model.addVar(lb=0, ub=max_B, name="x_B")
x_C = model.addVar(lb=0, ub=max_C, name="x_C")

delta_A = model.addVar(vtype=gp.GRB.BINARY, name="delta_A")
delta_B = model.addVar(vtype=gp.GRB.BINARY, name="delta_B")
delta_C = model.addVar(vtype=gp.GRB.BINARY, name="delta_C")

E_A = model.addVar(lb=0, name="E_A")
E_B = model.addVar(lb=0, name="E_B")
E_C = model.addVar(lb=0, name="E_C")

y_A = model.addVar(lb=0, ub=gp.GRB.INFINITY, name="y_A")
y_B = model.addVar(lb=0, ub=gp.GRB.INFINITY, name="y_B")
y_C = model.addVar(lb=0, ub=gp.GRB.INFINITY, name="y_C")

# Auxiliary variables for total investment per channel
s_A = model.addVar(lb=0, ub=gp.GRB.INFINITY, name="s_A")
s_B = model.addVar(lb=0, ub=gp.GRB.INFINITY, name="s_B")
s_C = model.addVar(lb=0, ub=gp.GRB.INFINITY, name="s_C")

# Auxiliary variables for square roots
t_A = model.addVar(lb=0, ub=gp.GRB.INFINITY, name="t_A")
t_B = model.addVar(lb=0, ub=gp.GRB.INFINITY, name="t_B")
t_C = model.addVar(lb=0, ub=gp.GRB.INFINITY, name="t_C")

# Auxiliary variables for products p_i = x_i * delta_i
p_A = model.addVar(lb=0, ub=gp.GRB.INFINITY, name="p_A")
p_B = model.addVar(lb=0, ub=gp.GRB.INFINITY, name="p_B")
p_C = model.addVar(lb=0, ub=gp.GRB.INFINITY, name="p_C")

# Set objective
model.setObjective(y_A + y_B + y_C, gp.GRB.MAXIMIZE)

# Constraints
# C1: Primary budget limit
model.addConstr(x_A + x_B + x_C <= total_budget, "PrimaryBudget")

# C2: Channel-wise upper bounds (already in variable bounds)

# C3: Exactly one channel is minimum
model.addConstr(delta_A + delta_B + delta_C == 1, "OneMinChannel")

# C4: Minimum-channel identification using indicator constraints
# If delta_A = 1, then x_A <= x_B and x_A <= x_C
model.addGenConstrIndicator(delta_A, 1, x_A <= x_B, "Min_A_le_B")
model.addGenConstrIndicator(delta_A, 1, x_A <= x_C, "Min_A_le_C")

# If delta_B = 1, then x_B <= x_A and x_B <= x_C
model.addGenConstrIndicator(delta_B, 1, x_B <= x_A, "Min_B_le_A")
model.addGenConstrIndicator(delta_B, 1, x_B <= x_C, "Min_B_le_C")

# If delta_C = 1, then x_C <= x_A and x_C <= x_B
model.addGenConstrIndicator(delta_C, 1, x_C <= x_A, "Min_C_le_A")
model.addGenConstrIndicator(delta_C, 1, x_C <= x_B, "Min_C_le_B")

# C5: Secondary investment definition
# p_A = x_A * delta_A
model.addGenConstrIndicator(delta_A, 1, p_A == x_A, "pA_if_deltaA_1")
model.addGenConstrIndicator(delta_A, 0, p_A == 0, "pA_if_deltaA_0")
model.addConstr(E_A == expansion_factor * p_A, "E_A_def")

model.addGenConstrIndicator(delta_B, 1, p_B == x_B, "pB_if_deltaB_1")
model.addGenConstrIndicator(delta_B, 0, p_B == 0, "pB_if_deltaB_0")
model.addConstr(E_B == expansion_factor * p_B, "E_B_def")

model.addGenConstrIndicator(delta_C, 1, p_C == x_C, "pC_if_deltaC_1")
model.addGenConstrIndicator(delta_C, 0, p_C == 0, "pC_if_deltaC_0")
model.addConstr(E_C == expansion_factor * p_C, "E_C_def")

# C6: Total investment per channel
model.addConstr(s_A == x_A + E_A, "s_A_def")
model.addConstr(s_B == x_B + E_B, "s_B_def")
model.addConstr(s_C == x_C + E_C, "s_C_def")

# C7: Total budget limit (including secondary)
model.addConstr(x_A + x_B + x_C + E_A + E_B + E_C <= total_budget, "TotalBudget")

# C8: Square root relationships
model.addGenConstrPow(t_A, s_A, 2, "sqrt_A")
model.addGenConstrPow(t_B, s_B, 2, "sqrt_B")
model.addGenConstrPow(t_C, s_C, 2, "sqrt_C")

# C9: Revenue functions
model.addConstr(y_A == coeff_A * t_A, "y_A_def")
model.addConstr(y_B == coeff_B * t_B, "y_B_def")
model.addConstr(y_C == coeff_C * t_C, "y_C_def")

# Solve the model
model.optimize()

# Print results
if model.status == gp.GRB.OPTIMAL:
    print("Optimal solution found")
    print(f"Primary investments: x_A = {x_A.X:.2f}, x_B = {x_B.X:.2f}, x_C = {x_C.X:.2f}")
    print(f"Minimum channel indicator: δ_A = {delta_A.X}, δ_B = {delta_B.X}, δ_C = {delta_C.X}")
    print(f"Secondary investments: E_A = {E_A.X:.2f}, E_B = {E_B.X:.2f}, E_C = {E_C.X:.2f}")
    print(f"Total investments: s_A = {s_A.X:.2f}, s_B = {s_B.X:.2f}, s_C = {s_C.X:.2f}")
    print(f"Revenues: y_A = {y_A.X:.2f}, y_B = {y_B.X:.2f}, y_C = {y_C.X:.2f}")
    print(f"Total revenue (objective): {model.ObjVal:.2f}")
    
    # Output final answer as required
    print(f"FinalAnswer=【{model.ObjVal:.2f}】")
else:
    print(f"Optimization failed. Status: {model.status}")
    print(f"FinalAnswer=【0】")