import gurobipy as gp

# ==================== Parameters from Parameters List ====================
supply_A = 6
supply_B = 8
procurement_cost_A = 9900  # (yuan/ton) - not directly used in final simplified objective
procurement_cost_B = 6600  # (yuan/ton) - not directly used in final simplified objective
consumption = {'A': {'I': 1, 'II': 2}, 'B': {'I': 2, 'II': 1}}
unit_price_material = {'A': 9.9, 'B': 6.6}  # thousand yuan/ton
wholesale_price = {'I': 30, 'II': 20}       # thousand yuan/thousand boxes
demand_upper_II = 2000
demand_diff_limit = 1000
demand_growth = 0.25

# ==================== Create Model ====================
model = gp.Model("TiantianFoodTwoMonthProfit")

# ==================== Decision Variables ====================
# Month 1 production/sales (thousand boxes)
x1 = model.addVar(lb=0, name="x1")  # Product I
x2 = model.addVar(lb=0, name="x2")  # Product II

# Month 2 production/sales (thousand boxes)
y1 = model.addVar(lb=0, name="y1")  # Product I
y2 = model.addVar(lb=0, name="y2")  # Product II

# ==================== Objective Function ====================
# Total two-month profit = Month1 wholesale revenue + Month2 surplus
# Month1 wholesale revenue: 30*x1 + 20*x2 (thousand yuan)
# Month2 surplus simplified: 6.9*y1 - 6.4*y2 (thousand yuan)
# Total in thousand yuan
model.setObjective(30*x1 + 20*x2 + 6.9*y1 - 6.4*y2, sense=gp.GRB.MAXIMIZE)

# ==================== Constraints ====================
# Month 1 constraints
# Raw material A: 1*x1 + 2*x2 ≤ 6 (tons)
model.addConstr(1*x1 + 2*x2 <= supply_A, name="month1_raw_A")

# Raw material B: 2*x1 + 1*x2 ≤ 8 (tons)
model.addConstr(2*x1 + 1*x2 <= supply_B, name="month1_raw_B")

# Demand cap for Product II: x2 ≤ 2 (thousand boxes)
model.addConstr(x2 <= demand_upper_II/1000, name="month1_demand_cap_II")

# Product II not exceed Product I by more than 1000 boxes: x2 - x1 ≤ 1
model.addConstr(x2 - x1 <= demand_diff_limit/1000, name="month1_II_vs_I_ratio")

# Month 2 constraints
# Demand cap for Product II grows by 25%: y2 ≤ 2.5
model.addConstr(y2 <= demand_upper_II/1000 * (1 + demand_growth), name="month2_demand_cap_II")

# Product II not exceed Product I by more than 1000 boxes: y2 - y1 ≤ 1
model.addConstr(y2 - y1 <= demand_diff_limit/1000, name="month2_II_vs_I_ratio")

# Note: No raw material constraints for month 2 as supply is sufficient

# ==================== Solve ====================
model.optimize()

# ==================== Results ====================
if model.status == gp.GRB.OPTIMAL:
    x1_val = x1.X
    x2_val = x2.X
    y1_val = y1.X
    y2_val = y2.X
    
    month1_revenue = 30*x1_val + 20*x2_val
    month2_surplus = 6.9*y1_val - 6.4*y2_val
    total_profit = month1_revenue + month2_surplus
    
    print("Optimal Solution Found:")
    print(f"Month 1 - Product I: {x1_val:.3f} thousand boxes")
    print(f"Month 1 - Product II: {x2_val:.3f} thousand boxes")
    print(f"Month 1 wholesale revenue: {month1_revenue:.3f} thousand yuan")
    print()
    print(f"Month 2 - Product I: {y1_val:.3f} thousand boxes")
    print(f"Month 2 - Product II: {y2_val:.3f} thousand boxes")
    print(f"Month 2 surplus: {month2_surplus:.3f} thousand yuan")
    print()
    print(f"Total two-month profit: {total_profit:.3f} thousand yuan")
    print(f"Total profit in yuan: {total_profit*1000:.2f}")
    
    # Output the final answer as requested
    print(f"FinalAnswer=【{total_profit*1000:.2f}】")
else:
    print("No optimal solution found.")
    print(f"FinalAnswer=【0】")