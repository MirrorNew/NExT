import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters (from Parameters List)
# =========================
factory_names = ['Foshan', 'Wuhu']
market_names = ['East China', 'South China']
audit_year = 2023
operational_data_year = 2024
num_factories = 2
num_markets = 2
demand = [80, 70]
a1 = 0.06
b1 = 11.0
p1_exp = 2.5
a2 = 0.03
b2 = 12.0
p2_exp = 1.5
transportation_costs = [[2.4, 0.5],
                        [4.0, 1.8]]
capacity = [100, 100]

# =========================
# 2. Create model
# =========================
model = gp.Model("TwoFactoryTwoMarket_NLP")

# Allow nonconvex power expressions
model.Params.NonConvex = 2

# =========================
# 3. Decision variables
# =========================
# Shipment variables x_ij: units shipped from factory i to market j
# i in {0,1} -> factories (0:Foshan, 1:Wuhu)
# j in {0,1} -> markets   (0:East China, 1:South China)
x = model.addVars(num_factories, num_markets, lb=0.0, vtype=GRB.CONTINUOUS, name="x")

# Production variables y_i: total output of each factory
y = model.addVars(num_factories, lb=0.0, ub=capacity[0], vtype=GRB.CONTINUOUS, name="y")
# Note: Both factories have same capacity=100 as given

# =========================
# 4. Auxiliary substitution variables
# =========================
# p1_var represents y1^2.5, p2_var represents y2^1.5
p1_var = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="p1_var")
p2_var = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="p2_var")

# c1_cost and c2_cost are the production costs of factory 1 and 2
c1_cost = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="c1_cost")
c2_cost = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="c2_cost")

# =========================
# 5. Objective function (will be set after constraints link auxiliaries)
# =========================
# Placeholder; we will set it after defining power constraints and linking costs

# =========================
# 6. Constraints
# =========================

# 6.1 Demand satisfaction constraints
# Market 1 (index 0): x11 + x21 = 80
# Market 2 (index 1): x12 + x22 = 70
for j in range(num_markets):
    model.addConstr(
        gp.quicksum(x[i, j] for i in range(num_factories)) == demand[j],
        name=f"Demand_Market{j+1}"
    )

# 6.2 Flow conservation constraints (link production and shipments)
# Factory 1: y1 - x11 - x12 = 0
# Factory 2: y2 - x21 - x22 = 0
for i in range(num_factories):
    model.addConstr(
        y[i] - gp.quicksum(x[i, j] for j in range(num_markets)) == 0,
        name=f"Flow_Conservation_Factory{i+1}"
    )

# 6.3 Capacity constraints
# Factory 1: y1 <= 100
# Factory 2: y2 <= 100
for i in range(num_factories):
    model.addConstr(y[i] <= capacity[i], name=f"Capacity_Factory{i+1}")

# 6.4 Nonlinear production cost modeling using auxiliary variables
# Power constraints:
# p1_var = y1^2.5, p2_var = y2^1.5
model.addGenConstrPow(y[0], p1_var, p1_exp, name="Pow_y1_2p5")
model.addGenConstrPow(y[1], p2_var, p2_exp, name="Pow_y2_1p5")

# Link c1_cost and c2_cost linearly with y and p
# c1_cost = 0.06 * p1_var + 11 * y1
model.addConstr(
    c1_cost == a1 * p1_var + b1 * y[0],
    name="ProdCost_Factory1"
)

# c2_cost = 0.03 * p2_var + 12 * y2
model.addConstr(
    c2_cost == a2 * p2_var + b2 * y[1],
    name="ProdCost_Factory2"
)

# =========================
# 5 (continued). Set objective: minimize total cost
# Z = c1_cost + c2_cost + sum_ij (transportation_costs[i][j] * x_ij)
# =========================
transport_cost_expr = gp.quicksum(
    transportation_costs[i][j] * x[i, j]
    for i in range(num_factories)
    for j in range(num_markets)
)

model.setObjective(c1_cost + c2_cost + transport_cost_expr, GRB.MINIMIZE)

# =========================
# 7. Solve model
# =========================
model.optimize()

# =========================
# 8. Print results
# =========================
if model.status == GRB.OPTIMAL or model.status == GRB.INTERRUPTED and model.SolCount > 0:
    print("\nOptimal solution found:")
    print(f"Objective value (Total cost): {model.ObjVal:.6f}")

    # Production levels
    for i in range(num_factories):
        print(f"Production y{i+1} ({factory_names[i]}): {y[i].X:.6f}")

    # Shipments
    for i in range(num_factories):
        for j in range(num_markets):
            print(f"x{i+1}{j+1} (Factory {factory_names[i]} -> Market {market_names[j]}): {x[i, j].X:.6f}")

    print(f"Production cost factory 1: {c1_cost.X:.6f}")
    print(f"Production cost factory 2: {c2_cost.X:.6f}")
    print(f"Aux p1_var (y1^{p1_exp}): {p1_var.X:.6f}")
    print(f"Aux p2_var (y2^{p2_exp}): {p2_var.X:.6f}")

    # Final answer is the optimal total cost
    final_answer = model.ObjVal
else:
    print("No optimal solution found.")
    final_answer = float('nan')

# =========================
# 9. Required final output line
# =========================
print(f"FinalAnswer=【{final_answer}】")