import gurobipy as gp
from gurobipy import GRB

# 2. Define all parameter matrices and data inputs
factory_names = ['Foshan', 'Wuhu']
market_names = ['East China', 'South China']
demand = [80, 70]
a1, b1, p1 = 0.06, 11.0, 2.5
a2, b2, p2 = 0.03, 12.0, 1.5
transportation_costs = [[2.4, 0.5], [4.0, 1.8]]
capacity = [100, 100]

# Initialize model
model = gp.Model("Supply_Chain_Optimization")

# Set parameter for non-convex general constraints
model.Params.NonConvex = 2

# 3. Create decision variables
x11 = model.addVar(lb=0, name="x11")
x12 = model.addVar(lb=0, name="x12")
x21 = model.addVar(lb=0, name="x21")
x22 = model.addVar(lb=0, name="x22")
y1 = model.addVar(lb=0, ub=capacity[0], name="y1")
y2 = model.addVar(lb=0, ub=capacity[1], name="y2")

# 4. Create auxiliary substitution variables
# These variables represent the power terms y1^2.5 and y2^1.5
v1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="v1")
v2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="v2")

# 5. Set up the objective function
# Total Cost = Production Cost (Factory 1 + Factory 2) + Transportation Costs
production_cost_f1 = a1 * v1 + b1 * y1
production_cost_f2 = a2 * v2 + b2 * y2
transportation_cost_total = (transportation_costs[0][0] * x11 + 
                             transportation_costs[0][1] * x12 + 
                             transportation_costs[1][0] * x21 + 
                             transportation_costs[1][1] * x22)

model.setObjective(production_cost_f1 + production_cost_f2 + transportation_cost_total, GRB.MINIMIZE)

# 6. Add all constraints
# General power constraints for auxiliary variables
# addGenConstrPow(x, y, a) implies y = x^a
model.addGenConstrPow(y1, v1, p1, name="v1_pow")
model.addGenConstrPow(y2, v2, p2, name="v2_pow")

# Demand fulfillment for Market 1
model.addConstr(x11 + x21 == demand[0], name="Demand_Market1")

# Demand fulfillment for Market 2
model.addConstr(x12 + x22 == demand[1], name="Demand_Market2")

# Flow conservation for Factory 1 (total output y1)
model.addConstr(y1 == x11 + x12, name="Flow_Conservation_Factory1")

# Flow conservation for Factory 2 (total output y2)
model.addConstr(y2 == x21 + x22, name="Flow_Conservation_Factory2")

# Production capacity constraints (redundant if using ub on y1 and y2, but kept for clarity)
model.addConstr(y1 <= capacity[0], name="Capacity_Factory1")
model.addConstr(y2 <= capacity[1], name="Capacity_Factory2")

# 7. Solve the model and print results
model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"FinalAnswer=【{model.objVal}】")