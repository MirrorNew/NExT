import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs
factory_names = ['Foshan', 'Wuhu']
market_names = ['East China', 'South China']
audit_year = 2023
operational_data_year = 2024
num_factories = 2
num_markets = 2
demand = [80, 70]
a1 = 0.06
b1 = 11.0
p1 = 2.5
a2 = 0.03
b2 = 12.0
p2 = 1.5
transportation_costs = [[2.4, 0.5], [4.0, 1.8]]
capacity = [100, 100]

# 2. Create Gurobi Model
model = gp.Model("SupplyChainOptimization")

# Enable NonConvex parameter for general power constraints if needed
model.Params.NonConvex = 2

# 3. Create decision variables
# x[i,j]: Quantity shipped from factory i to market j
x = model.addVars(num_factories, num_markets, lb=0, vtype=GRB.CONTINUOUS, name="x")

# y[i]: Total output of factory i
y = model.addVars(num_factories, lb=0, vtype=GRB.CONTINUOUS, name="y")

# 4. Create auxiliary substitution variables
# aux_y_pow[i] will store the value of the nonlinear term for factory i
# For Factory 1: y[0]^2.5
# For Factory 2: y[1]^1.5
# As per instructions, auxiliary variables have infinite bounds
aux_y_pow = model.addVars(num_factories, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="aux_y_pow")

# 5. Set up the objective function
# Total Cost = Production Cost + Transportation Cost
# Factory 1 Cost: a1 * y1^p1 + b1 * y1 -> a1 * aux_y_pow[0] + b1 * y[0]
# Factory 2 Cost: a2 * y2^p2 + b2 * y2 -> a2 * aux_y_pow[1] + b2 * y[1]
production_cost = (a1 * aux_y_pow[0] + b1 * y[0]) + (a2 * aux_y_pow[1] + b2 * y[1])

# Transportation Cost: sum(cost[i][j] * x[i,j])
transportation_cost = gp.quicksum(transportation_costs[i][j] * x[i, j] 
                                  for i in range(num_factories) 
                                  for j in range(num_markets))

model.setObjective(production_cost + transportation_cost, GRB.MINIMIZE)

# 6. Add all constraints

# (1) Nonlinear Power Constraints using General Constraints
# y[0]^p1 = aux_y_pow[0] => y[0]^2.5
model.addGenConstrPow(y[0], aux_y_pow[0], p1, name="Power_Factory1")
# y[1]^p2 = aux_y_pow[1] => y[1]^1.5
model.addGenConstrPow(y[1], aux_y_pow[1], p2, name="Power_Factory2")

# (2) Demand Satisfaction Constraints
# Market 1 (index 0)
model.addConstr(gp.quicksum(x[i, 0] for i in range(num_factories)) == demand[0], name="Demand_Market1")
# Market 2 (index 1)
model.addConstr(gp.quicksum(x[i, 1] for i in range(num_factories)) == demand[1], name="Demand_Market2")

# (3) Flow Conservation (Production Definition)
# Output of factory i must equal sum of shipments from factory i
for i in range(num_factories):
    model.addConstr(y[i] == gp.quicksum(x[i, j] for j in range(num_markets)), name=f"Flow_Factory{i+1}")

# (4) Production Capacity Constraints
for i in range(num_factories):
    model.addConstr(y[i] <= capacity[i], name=f"Capacity_Factory{i+1}")

# 7. Solve the model and print results
model.optimize()

if model.Status == GRB.OPTIMAL:
    print("\nOptimal Solution Found:")
    print(f"Total Cost: {model.ObjVal}")
    for v in model.getVars():
        if v.X > 1e-6: # Print only non-zero variables for clarity
            print(f"{v.VarName}: {v.X}")
    
    # Output the final answer in the required format
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("No optimal solution found.")