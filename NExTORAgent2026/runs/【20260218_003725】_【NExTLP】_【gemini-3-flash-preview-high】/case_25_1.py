import gurobipy as gp
from gurobipy import GRB

# Define the parameters
supply_A_m1 = 6  # tons
supply_B_m1 = 8  # tons
procurement_cost_A = 9.9  # thousand yuan/ton
procurement_cost_B = 6.6  # thousand yuan/ton
wholesale_price_I = 30  # thousand yuan/thousand boxes
wholesale_price_II = 20  # thousand yuan/thousand boxes
demand_upper_II_m1 = 2.0  # thousand boxes (given 2000 boxes)
demand_diff_limit_m1 = 1.0  # thousand boxes (given 1000 boxes)
demand_growth = 0.25

# Create a model for Month 1
model1 = gp.Model("Month1")

# Decision variables for Month 1
x1 = model1.addVar(lb=0, name="x1")  # Product I (thousand boxes)
x2 = model1.addVar(lb=0, name="x2")  # Product II (thousand boxes)

# Objective function for Month 1: maximize wholesale revenue
# Z1 = 30*x1 + 20*x2
model1.setObjective(wholesale_price_I * x1 + wholesale_price_II * x2, GRB.MAXIMIZE)

# Constraints for Month 1
# Raw material constraints
model1.addConstr(1 * x1 + 2 * x2 <= supply_A_m1, name="Raw_A_limit_m1")
model1.addConstr(2 * x1 + 1 * x2 <= supply_B_m1, name="Raw_B_limit_m1")
# Demand constraints
model1.addConstr(x2 <= demand_upper_II_m1, name="Demand_cap_II_m1")
model1.addConstr(x2 - x1 <= demand_diff_limit_m1, name="II_vs_I_ratio_m1")

# Solve Month 1 model
model1.optimize()

# Extract results for Month 1
x1_opt = x1.X
x2_opt = x2.X
revenue1 = model1.ObjVal
# Month 1 profit is revenue minus material expenditure
cost1 = (procurement_cost_A * (1 * x1_opt + 2 * x2_opt) + 
         procurement_cost_B * (2 * x1_opt + 1 * x2_opt))
profit1 = revenue1 - cost1

# Prepare for Month 2
# In Month 2, material supply is "sufficient" (not a limiting factor)
# The overall market demand grows by 25%. This applies to demand caps.
# The implicit demand cap for Product I in Month 1 was 4.0, which was the 
# most restrictive material constraint for Product I (2*x1 <= 8).
demand_upper_I_m1 = 4.0 
demand_upper_I_m2 = demand_upper_I_m1 * (1 + demand_growth)  # 4 * 1.25 = 5.0
demand_upper_II_m2 = demand_upper_II_m1 * (1 + demand_growth)  # 2 * 1.25 = 2.5

# Create a model for Month 2
model2 = gp.Model("Month2")

# Decision variables for Month 2
y1 = model2.addVar(lb=0, name="y1")  # Product I (thousand boxes)
y2 = model2.addVar(lb=0, name="y2")  # Product II (thousand boxes)

# Objective function for Month 2: maximize surplus (wholesale revenue minus material expenditure)
# surplus = 30*y1 + 20*y2 - [9.9*(1*y1 + 2*y2) + 6.6*(2*y1 + 1*y2)]
# simplified: surplus = 6.9*y1 - 6.4*y2
surplus_m2 = (wholesale_price_I * y1 + wholesale_price_II * y2) - \
             (procurement_cost_A * (1 * y1 + 2 * y2) + \
              procurement_cost_B * (2 * y1 + 1 * y2))
model2.setObjective(surplus_m2, GRB.MAXIMIZE)

# Constraints for Month 2
# Market growth limits
model2.addConstr(y1 <= demand_upper_I_m2, name="Demand_cap_I_m2")
model2.addConstr(y2 <= demand_upper_II_m2, name="Demand_cap_II_m2")
# Ratio constraint remains (II vs I ratio as per the context)
model2.addConstr(y2 - y1 <= demand_diff_limit_m1, name="II_vs_I_ratio_m2")

# Solve Month 2 model
model2.optimize()

# Extract results for Month 2
profit2 = model2.ObjVal

# Total Profit for the two months
total_profit = profit1 + profit2

# Print results
print(f"Optimal Production Month 1: Product I = {x1_opt:.4f}, Product II = {x2_opt:.4f}")
print(f"Profit Month 1: {profit1:.4f}")
print(f"Optimal Production Month 2: Product I = {y1.X:.4f}, Product II = {y2.X:.4f}")
print(f"Profit Month 2: {profit2:.4f}")
print(f"FinalAnswer=【{total_profit}】")