import gurobipy as gp

# Create model
model = gp.Model("SupplyChainOptimization")

# Set non-convex mode for power functions
model.Params.NonConvex = 2

# 1. Parameters
# Factory and market names (for reference)
factory_names = ['Foshan', 'Wuhu']
market_names = ['East China', 'South China']

# Demands
demand = [80, 70]  # market 1: 80, market 2: 70

# Production cost function parameters
a1 = 0.06
b1 = 11.0
p1 = 2.5  # exponent for y1^2.5

a2 = 0.03
b2 = 12.0
p2 = 1.5  # exponent for y2^1.5

# Transportation costs: [factory][market]
transportation_costs = [[2.4, 0.5], [4.0, 1.8]]

# Capacities
capacity = [100, 100]

# 2. Decision variables
x11 = model.addVar(lb=0, name="x11")  # factory 1 -> market 1
x12 = model.addVar(lb=0, name="x12")  # factory 1 -> market 2
x21 = model.addVar(lb=0, name="x21")  # factory 2 -> market 1
x22 = model.addVar(lb=0, name="x22")  # factory 2 -> market 2

y1 = model.addVar(lb=0, ub=capacity[0], name="y1")  # output factory 1
y2 = model.addVar(lb=0, ub=capacity[1], name="y2")  # output factory 2

# 3. Auxiliary variables for nonlinear terms
Z1 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="Z1")  # for y1^2.5
Z2 = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="Z2")  # for y2^1.5

# 4. Constraints
# Demand constraints
model.addConstr(x11 + x21 == demand[0], "Demand_Market1")
model.addConstr(x12 + x22 == demand[1], "Demand_Market2")

# Flow conservation
model.addConstr(y1 == x11 + x12, "Flow_Conservation_Factory1")
model.addConstr(y2 == x21 + x22, "Flow_Conservation_Factory2")

# Capacity constraints (already handled by ub on y1, y2)
# model.addConstr(y1 <= capacity[0], "Capacity_Factory1")  # redundant
# model.addConstr(y2 <= capacity[1], "Capacity_Factory2")  # redundant

# 5. General constraints for power functions
model.addGenConstrPow(y1, Z1, p1, "Power_y1")  # Z1 = y1^2.5
model.addGenConstrPow(y2, Z2, p2, "Power_y2")  # Z2 = y2^1.5

# 6. Objective function
production_cost = a1 * Z1 + b1 * y1 + a2 * Z2 + b2 * y2
transport_cost = transportation_costs[0][0] * x11 + transportation_costs[0][1] * x12 + \
                 transportation_costs[1][0] * x21 + transportation_costs[1][1] * x22

total_cost = production_cost + transport_cost
model.setObjective(total_cost, gp.GRB.MINIMIZE)

# 7. Solve
model.optimize()

# 8. Print results
if model.status == gp.GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"Minimum total cost: {model.ObjVal:.2f}")
    print("\nShipments:")
    print(f"x11 (Foshan -> East China): {x11.X:.2f}")
    print(f"x12 (Foshan -> South China): {x12.X:.2f}")
    print(f"x21 (Wuhu -> East China): {x21.X:.2f}")
    print(f"x22 (Wuhu -> South China): {x22.X:.2f}")
    print(f"\nProduction:")
    print(f"y1 (Foshan output): {y1.X:.2f}")
    print(f"y2 (Wuhu output): {y2.X:.2f}")
    print(f"\nVerification:")
    print(f"Market 1 demand: {x11.X + x21.X:.2f} (required: {demand[0]})")
    print(f"Market 2 demand: {x12.X + x22.X:.2f} (required: {demand[1]})")
    print(f"Factory 1 flow balance: {y1.X:.2f} = {x11.X + x12.X:.2f}")
    print(f"Factory 2 flow balance: {y2.X:.2f} = {x21.X + x22.X:.2f}")
    
    # Final answer - the minimum total cost
    print(f"FinalAnswer=【{model.ObjVal:.2f}】")
else:
    print("No optimal solution found.")
    print(f"FinalAnswer=【No optimal solution】")