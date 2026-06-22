import gurobipy as gp

# Parameters from the list
delivery_quantities = [40, 60, 80]
max_capacity = 100
cost_a = 50
cost_b = 0.002
cost_exp = 2.9
storage_cost = 4
initial_inventory = 0

# Create model
model = gp.Model("EngineProductionInventory")

# Decision variables
x = {}
y = {}  # Auxiliary: y_t = x_t^2.9
p = {}  # Auxiliary: p_t = 0.002 * y_t (nonlinear production cost component)
I = {}  # Inventory at end of quarter t

# Time periods (quarters)
T = 3

# Create variables
for t in range(1, T+1):
    # Production quantity (integer)
    x[t] = model.addVar(lb=0, ub=max_capacity, vtype=gp.GRB.INTEGER, name=f"x_{t}")
    
    # Auxiliary variable for x_t^2.9 (unbounded)
    y[t] = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name=f"y_{t}")
    
    # Auxiliary variable for nonlinear cost component (unbounded)
    p[t] = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name=f"p_{t}")
    
    # Inventory at end of quarter t (integer, non-negative)
    I[t] = model.addVar(lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.INTEGER, name=f"I_{t}")

# Initial inventory variable (for completeness)
I[0] = model.addVar(lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.INTEGER, name=f"I_0")

# Set initial inventory constraint
model.addConstr(I[0] == initial_inventory, "initial_inventory")

# Set NonConvex parameter for power constraints
model.Params.NonConvex = 2

# Add constraints
for t in range(1, T+1):
    # Power constraint: y_t = x_t^2.9
    model.addGenConstrPow(x[t], y[t], cost_exp, f"power_constr_{t}")
    
    # Linear constraint for nonlinear cost: p_t = 0.002 * y_t
    model.addConstr(p[t] == cost_b * y[t], f"nonlinear_cost_constr_{t}")
    
    # Inventory balance constraints
    if t == 1:
        model.addConstr(I[1] == I[0] + x[1] - delivery_quantities[0], "inventory_balance_1")
    elif t == 2:
        model.addConstr(I[2] == I[1] + x[2] - delivery_quantities[1], "inventory_balance_2")
    else:  # t == 3
        model.addConstr(I[3] == I[2] + x[3] - delivery_quantities[2], "inventory_balance_3")

# Set objective function
obj_expr = gp.quicksum(cost_a * x[t] + p[t] for t in range(1, T+1))
obj_expr += gp.quicksum(storage_cost * I[t] for t in range(1, T+1))
model.setObjective(obj_expr, gp.GRB.MINIMIZE)

# Optimize the model
model.optimize()

# Print solution
if model.status == gp.GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"Total minimum cost: {model.objVal:.2f} yuan")
    print("\nProduction plan:")
    for t in range(1, T+1):
        print(f"Quarter {t}: Produce {x[t].X:.0f} engines")
    print("\nEnding inventory:")
    for t in range(1, T+1):
        print(f"End of quarter {t}: {I[t].X:.0f} engines")
    
    # Calculate total cost breakdown
    production_cost = sum(cost_a * x[t].X + cost_b * (x[t].X ** cost_exp) for t in range(1, T+1))
    storage_cost_total = sum(storage_cost * I[t].X for t in range(1, T+1))
    print(f"\nCost breakdown:")
    print(f"  Production cost: {production_cost:.2f} yuan")
    print(f"  Storage cost: {storage_cost_total:.2f} yuan")
    print(f"  Total cost: {production_cost + storage_cost_total:.2f} yuan")
    
    # Final answer as required
    print(f"\nFinalAnswer=【{model.objVal:.2f}】")
else:
    print(f"Optimization failed. Status: {model.status}")
    print("FinalAnswer=【None】")