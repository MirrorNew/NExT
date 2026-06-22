import gurobipy as gp
from gurobipy import GRB

# Define the model
model = gp.Model("Tiantian_Food_Production")

# 1. Define Parameters
supply_A_m1 = 6
supply_B_m1 = 8
cost_A = 9.9
cost_B = 6.6
price_I = 30
price_II = 20
consumption = {'A': {'I': 1, 'II': 2}, 'B': {'I': 2, 'II': 1}}
demand_cap_II_m1 = 2.0  # 2000 boxes -> 2.0
demand_diff_limit = 1.0 # 1000 boxes -> 1.0
growth_rate = 0.25

# 2. Create Decision Variables for Month 1
# x1: Product I in Month 1, x2: Product II in Month 1
x1 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="x1")
x2 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="x2")

# 3. Set up Month 1 Constraints
# Raw A limit (month 1)
model.addConstr(consumption['A']['I']*x1 + consumption['A']['II']*x2 <= supply_A_m1, 
                name="Raw A limit (month 1)")

# Raw B limit (month 1)
model.addConstr(consumption['B']['I']*x1 + consumption['B']['II']*x2 <= supply_B_m1, 
                name="Raw B limit (month 1)")

# Demand cap II (month 1)
model.addConstr(x2 <= demand_cap_II_m1, name="Demand cap II (month 1)")

# II vs I ratio (month 1)
model.addConstr(x2 - x1 <= demand_diff_limit, name="II vs I ratio (month 1)")

# 4. Set Month 1 Objective: Maximize Wholesale Revenue
model.setObjective(price_I*x1 + price_II*x2, GRB.MAXIMIZE)

# 5. Solve Month 1
model.optimize()

if model.Status != GRB.OPTIMAL:
    print("Optimization for Month 1 failed.")
else:
    # Retrieve Month 1 results
    x1_val = x1.X
    x2_val = x2.X
    revenue_m1 = model.ObjVal
    
    # Calculate Month 1 Cost and Profit
    # Cost = Cost_A * Usage_A + Cost_B * Usage_B
    usage_A_m1 = consumption['A']['I']*x1_val + consumption['A']['II']*x2_val
    usage_B_m1 = consumption['B']['I']*x1_val + consumption['B']['II']*x2_val
    cost_m1 = cost_A * usage_A_m1 + cost_B * usage_B_m1
    profit_m1 = revenue_m1 - cost_m1
    
    print(f"Month 1 Plan: x1={x1_val:.4f}, x2={x2_val:.4f}")
    print(f"Month 1 Revenue: {revenue_m1:.4f}, Cost: {cost_m1:.4f}, Profit: {profit_m1:.4f}")

    # 6. Set up Month 2
    # Fix Month 1 variables to preserve the plan
    x1.LB = x1_val
    x1.UB = x1_val
    x2.LB = x2_val
    x2.UB = x2_val

    # Create Decision Variables for Month 2
    y1 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="y1")
    y2 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="y2")
    
    # Calculate Demand Caps for Month 2
    # Demand II grows by 25% from the base cap
    demand_cap_II_m2 = demand_cap_II_m1 * (1 + growth_rate)
    
    # Demand I grows by 25% relative to the realized sales/capacity of Month 1
    # (Since Month 1 sales were supply-constrained but demand was "high", 
    # M2 demand is modeled as an expansion of that realized base).
    demand_cap_I_m2 = x1_val * (1 + growth_rate)
    
    # Month 2 Constraints
    # Demand cap II (month 2)
    model.addConstr(y2 <= demand_cap_II_m2, name="Demand cap II (month 2)")
    
    # II vs I ratio (month 2)
    model.addConstr(y2 - y1 <= demand_diff_limit, name="II vs I ratio (month 2)")
    
    # Demand cap I (month 2) - derived from growth constraint
    model.addConstr(y1 <= demand_cap_I_m2, name="Demand cap I (month 2)")
    
    # 7. Set Month 2 Objective: Maximize Surplus (Revenue - Cost)
    # Surplus = (Price - UnitCost)*y
    unit_cost_I = cost_A * consumption['A']['I'] + cost_B * consumption['B']['I']
    unit_cost_II = cost_A * consumption['A']['II'] + cost_B * consumption['B']['II']
    
    margin_I = price_I - unit_cost_I
    margin_II = price_II - unit_cost_II
    
    # Objective is purely Month 2 surplus (since M1 is fixed/sunk for this step)
    model.setObjective(margin_I*y1 + margin_II*y2, GRB.MAXIMIZE)
    
    # 8. Solve Month 2
    model.optimize()
    
    if model.Status == GRB.OPTIMAL:
        y1_val = y1.X
        y2_val = y2.X
        profit_m2 = model.ObjVal # This is exactly the surplus
        
        print(f"Month 2 Plan: y1={y1_val:.4f}, y2={y2_val:.4f}")
        print(f"Month 2 Profit (Surplus): {profit_m2:.4f}")
        
        # Calculate Total Profit
        total_profit = profit_m1 + profit_m2
        
        print(f"FinalAnswer=【{total_profit}】")
    else:
        print("Optimization for Month 2 failed.")