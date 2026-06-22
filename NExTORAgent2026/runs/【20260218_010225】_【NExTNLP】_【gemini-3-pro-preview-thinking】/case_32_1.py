import gurobipy as gp
from gurobipy import GRB

# 1. Create Model
model = gp.Model("RhodeIslandMediaOptimization")
model.Params.NonConvex = 2  # Enable handling for non-convex power constraints and logical constraints

# 2. Define Parameters
saturation_threshold_TV = 600000
total_budget = 100
max_investment = {'A': 60, 'B': 50, 'C': 50}
revenue_coefficients = {'A': 30.23, 'B': 24.36, 'C': 20.12}

# 3. Create Decision Variables

# Primary advertising investment variables
x_A = model.addVar(lb=0, ub=max_investment['A'], vtype=GRB.CONTINUOUS, name="x_A")
x_B = model.addVar(lb=0, ub=max_investment['B'], vtype=GRB.CONTINUOUS, name="x_B")
x_C = model.addVar(lb=0, ub=max_investment['C'], vtype=GRB.CONTINUOUS, name="x_C")

# Binary indicator variables for the lowest primary investment
delta_A = model.addVar(vtype=GRB.BINARY, name="delta_A")
delta_B = model.addVar(vtype=GRB.BINARY, name="delta_B")
delta_C = model.addVar(vtype=GRB.BINARY, name="delta_C")

# Secondary (effect-amplifier) investment variables
E_A = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="E_A")
E_B = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="E_B")
E_C = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="E_C")

# Expected revenue variables
y_A = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y_A")
y_B = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y_B")
y_C = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y_C")

# 4. Auxiliary Substitution Variables

# Total investment (Primary + Secondary)
Total_A = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Total_A")
Total_B = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Total_B")
Total_C = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Total_C")

# Square root of total investment
Sqrt_A = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Sqrt_A")
Sqrt_B = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Sqrt_B")
Sqrt_C = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Sqrt_C")

# 5. Objective Function
model.setObjective(y_A + y_B + y_C, GRB.MAXIMIZE)

# 6. Add Constraints

# C1. TV saturation threshold
model.addConstr(x_A <= saturation_threshold_TV, "C1_TV_Saturation")

# C2. Initial budget limit
model.addConstr(x_A + x_B + x_C <= total_budget, "C2_Initial_Budget")

# C3, C4, C5 are handled by variable bounds (ub=max_investment), 
# but we can add them explicitly if strict adherence to constraint list is needed.
# Since bounds are set, these are redundant but safe.
model.addConstr(x_A <= 60, "C3_Max_A")
model.addConstr(x_B <= 50, "C4_Max_B")
model.addConstr(x_C <= 50, "C5_Max_C")

# C6. Binary‐sum for min detection
model.addConstr(delta_A + delta_B + delta_C == 1, "C6_One_Min_Channel")

# C7. Min‐channel identification, A (Indicator constraints instead of Big-M)
model.addGenConstrIndicator(delta_A, 1, x_A <= x_B, name="Ind_Min_A_vs_B")
model.addGenConstrIndicator(delta_A, 1, x_A <= x_C, name="Ind_Min_A_vs_C")

# C8. Min‐channel identification, B
model.addGenConstrIndicator(delta_B, 1, x_B <= x_A, name="Ind_Min_B_vs_A")
model.addGenConstrIndicator(delta_B, 1, x_B <= x_C, name="Ind_Min_B_vs_C")

# C9. Min‐channel identification, C
model.addGenConstrIndicator(delta_C, 1, x_C <= x_A, name="Ind_Min_C_vs_A")
model.addGenConstrIndicator(delta_C, 1, x_C <= x_B, name="Ind_Min_C_vs_B")

# C10. Secondary investment definition
# Channel A
model.addGenConstrIndicator(delta_A, 1, E_A == 3 * x_A, name="Ind_Sec_A_Active")
model.addGenConstrIndicator(delta_A, 0, E_A == 0, name="Ind_Sec_A_Inactive")
# Channel B
model.addGenConstrIndicator(delta_B, 1, E_B == 3 * x_B, name="Ind_Sec_B_Active")
model.addGenConstrIndicator(delta_B, 0, E_B == 0, name="Ind_Sec_B_Inactive")
# Channel C
model.addGenConstrIndicator(delta_C, 1, E_C == 3 * x_C, name="Ind_Sec_C_Active")
model.addGenConstrIndicator(delta_C, 0, E_C == 0, name="Ind_Sec_C_Inactive")

# C11. Final budget (incl. secondary)
model.addConstr(x_A + x_B + x_C + E_A + E_B + E_C <= total_budget, "C11_Final_Budget")

# C12. Revenue function definitions
# Define Total investment first
model.addConstr(Total_A == x_A + E_A, "Def_Total_A")
model.addConstr(Total_B == x_B + E_B, "Def_Total_B")
model.addConstr(Total_C == x_C + E_C, "Def_Total_C")

# Define Square Root relationships: Sqrt = Total^0.5
model.addGenConstrPow(Total_A, Sqrt_A, 0.5, "Pow_Sqrt_A")
model.addGenConstrPow(Total_B, Sqrt_B, 0.5, "Pow_Sqrt_B")
model.addGenConstrPow(Total_C, Sqrt_C, 0.5, "Pow_Sqrt_C")

# Calculate Revenue
model.addConstr(y_A == revenue_coefficients['A'] * Sqrt_A, "Def_y_A")
model.addConstr(y_B == revenue_coefficients['B'] * Sqrt_B, "Def_y_B")
model.addConstr(y_C == revenue_coefficients['C'] * Sqrt_C, "Def_y_C")

# 7. Solve and Print
model.optimize()

if model.Status == GRB.OPTIMAL:
    print(f"Optimal Objective Value: {model.ObjVal}")
    print(f"Investments: A={x_A.X}, B={x_B.X}, C={x_C.X}")
    print(f"Secondary: E_A={E_A.X}, E_B={E_B.X}, E_C={E_C.X}")
    print(f"Min Channel Indicators: A={delta_A.X}, B={delta_B.X}, C={delta_C.X}")
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimization unsuccessful")