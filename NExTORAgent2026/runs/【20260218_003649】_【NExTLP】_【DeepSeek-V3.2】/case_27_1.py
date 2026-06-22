import gurobipy as gp

# Create model
model = gp.Model("HongMudan_Production_Optimization")

# Parameters from Parameters List
total_tomatoes = 300000
grade_A_pct = 0.2
grade_B_pct = 0.7
grade_C_pct = 0.1
grade_C_initial_score = 1
grade_C_research_sale_price = 0.001
research_sales_threshold1 = 20000
research_success_prob1 = 0.8
research_post_score1 = 4
research_unsold_threshold = 10000
research_success_prob2 = 0.2
avg_score_A = 9
avg_score_B = 5
min_avg_score_whole = 8
min_avg_score_juice = 6
min_avg_score_sauce = 4

# Calculate available quantities
grade_A_available = total_tomatoes * grade_A_pct  # 60,000 kg
grade_B_available = total_tomatoes * grade_B_pct  # 210,000 kg
grade_C_available = total_tomatoes * grade_C_pct  # 30,000 kg

# Demand limits from Table C-10
demand_whole = 800000
demand_juice = 50000
demand_sauce = 80000

# Profit per can from Table C-11
profit_whole = 0.40
profit_juice = 0.12
profit_sauce = 0.35

# Decision variables
x_W = model.addVar(lb=0, ub=demand_whole, name="x_W")  # cans of whole tomatoes
x_J = model.addVar(lb=0, ub=demand_juice, name="x_J")   # cans of tomato juice
x_S = model.addVar(lb=0, ub=demand_sauce, name="x_S")   # cans of tomato sauce

# Tomato usage variables
y_AW = model.addVar(lb=0, ub=grade_A_available, name="y_AW")  # A for whole
y_BW = model.addVar(lb=0, ub=grade_B_available, name="y_BW")  # B for whole
y_AJ = model.addVar(lb=0, ub=grade_A_available, name="y_AJ")  # A for juice
y_BJ = model.addVar(lb=0, ub=grade_B_available, name="y_BJ")  # B for juice
y_AS = model.addVar(lb=0, ub=grade_A_available, name="y_AS")  # A for sauce
y_BS = model.addVar(lb=0, ub=grade_B_available, name="y_BS")  # B for sauce
y_CS = model.addVar(lb=0, ub=grade_C_available, name="y_CS")  # C for sauce
y_R = model.addVar(lb=0, ub=grade_C_available, name="y_R")    # C sold to research

# Binary variable for research threshold
z_R = model.addVar(vtype=gp.GRB.BINARY, name="z_R")

# Set objective
model.setObjective(
    profit_whole * x_W + profit_juice * x_J + profit_sauce * x_S + grade_C_research_sale_price * y_R,
    gp.GRB.MAXIMIZE
)

# Supply constraints
model.addConstr(y_AW + y_AJ + y_AS <= grade_A_available, "Grade_A_supply")
model.addConstr(y_BW + y_BJ + y_BS <= grade_B_available, "Grade_B_supply")
model.addConstr(y_CS + y_R <= grade_C_available, "Grade_C_balance")

# Raw material usage constraints
model.addConstr(y_AW + y_BW == 1.8 * x_W, "Raw_material_whole")
model.addConstr(y_AJ + y_BJ == 2.0 * x_J, "Raw_material_juice")
model.addConstr(y_AS + y_BS + y_CS == 2.5 * x_S, "Raw_material_sauce")

# Quality constraints
# Whole tomatoes: 9y_AW + 5y_BW >= 8(y_AW + y_BW) => y_AW >= 3y_BW
model.addConstr(y_AW >= 3 * y_BW, "Quality_whole")

# Tomato juice: 9y_AJ + 5y_BJ >= 6(y_AJ + y_BJ) => 3y_AJ >= y_BJ
model.addConstr(3 * y_AJ >= y_BJ, "Quality_juice")

# Tomato sauce quality constraint with R&D effect
# Effective C score = 1.6 + 1.8 * z_R
# Constraint: 9y_AS + 5y_BS + (1.6 + 1.8*z_R)*y_CS >= 4(y_AS + y_BS + y_CS)
# => 5y_AS + y_BS - 2.4y_CS + 1.8*z_R*y_CS >= 0

# Create auxiliary variable w = z_R * y_CS
w = model.addVar(lb=0, ub=grade_C_available, name="w")

# Linearize w = z_R * y_CS using indicator constraints
model.addGenConstrIndicator(z_R, 1, w == y_CS, name="w_eq_y_CS_if_zR1")
model.addGenConstrIndicator(z_R, 0, w == 0, name="w_eq_0_if_zR0")

# Add the sauce quality constraint
model.addConstr(5 * y_AS + y_BS - 2.4 * y_CS + 1.8 * w >= 0, "Quality_sauce")

# R&D threshold constraints using indicator constraints
M = grade_C_available  # Big M value

# If z_R = 1, then y_R >= 20000
model.addGenConstrIndicator(z_R, 1, y_R >= research_sales_threshold1, name="R&D_threshold_lower")

# If z_R = 0, then y_R <= 19999.999 (using epsilon to avoid strict inequality)
epsilon = 0.001
model.addGenConstrIndicator(z_R, 0, y_R <= research_sales_threshold1 - epsilon, name="R&D_threshold_upper")

# Solve the model
model.optimize()

# Print results
if model.status == gp.GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"Objective value (total profit): {model.objVal:.2f} yuan")
    print("\nProduction quantities:")
    print(f"  Whole tomatoes: {x_W.x:.0f} cans")
    print(f"  Tomato juice: {x_J.x:.0f} cans")
    print(f"  Tomato sauce: {x_S.x:.0f} cans")
    
    print("\nTomato usage (kg):")
    print(f"  Grade A for whole: {y_AW.x:.2f}")
    print(f"  Grade B for whole: {y_BW.x:.2f}")
    print(f"  Grade A for juice: {y_AJ.x:.2f}")
    print(f"  Grade B for juice: {y_BJ.x:.2f}")
    print(f"  Grade A for sauce: {y_AS.x:.2f}")
    print(f"  Grade B for sauce: {y_BS.x:.2f}")
    print(f"  Grade C for sauce: {y_CS.x:.2f}")
    print(f"  Grade C sold to research: {y_R.x:.2f}")
    
    print(f"\nR&D threshold indicator (z_R): {z_R.x:.0f}")
    
    # Calculate total tomato usage
    total_used = (y_AW.x + y_BW.x + y_AJ.x + y_BJ.x + y_AS.x + y_BS.x + y_CS.x + y_R.x)
    print(f"\nTotal tomatoes used: {total_used:.2f} kg out of {total_tomatoes} kg")
    
    # Calculate profit breakdown
    profit_whole_total = profit_whole * x_W.x
    profit_juice_total = profit_juice * x_J.x
    profit_sauce_total = profit_sauce * x_S.x
    profit_research = grade_C_research_sale_price * y_R.x
    
    print("\nProfit breakdown:")
    print(f"  Whole tomatoes: {profit_whole_total:.2f} yuan")
    print(f"  Tomato juice: {profit_juice_total:.2f} yuan")
    print(f"  Tomato sauce: {profit_sauce_total:.2f} yuan")
    print(f"  Research sales: {profit_research:.2f} yuan")
    
    # Output the final answer (maximum profit)
    print(f"\nFinalAnswer=【{model.objVal:.2f}】")
else:
    print("No optimal solution found")
    print(f"FinalAnswer=【0】")