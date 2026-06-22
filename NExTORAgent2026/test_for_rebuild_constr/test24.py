import gurobipy as gp

# Parameters from the provided list
standard_panel_length = 100
factory1_modes = [1, 2, 3]
factory2_modes = [4, 5]
demand_25 = 8
demand_40 = 6
demand_50 = 4

Table_1_CuttingCombination = [
    {'mode': 1, 'cutting_combination': [40, 25, 25], 'waste': 10},
    {'mode': 2, 'cutting_combination': [40, 40], 'waste': 20},
    {'mode': 3, 'cutting_combination': [50, 40], 'waste': 10},
    {'mode': 4, 'cutting_combination': [50, 25, 25], 'waste': 0},
    {'mode': 5, 'cutting_combination': [25, 25, 25, 25], 'waste': 0},
    {'mode': 6, 'cutting_combination': [50, 50], 'waste': 0}
]

Table_3_SyntheticWoodFurniture = [
    {'furniture': 'Bench', 'consumed_meters': 20, 'max_required_quantity': 12, 'profit': 3},
    {'furniture': 'Chair', 'consumed_meters': 40, 'max_required_quantity': 7,  'profit': 8},
    {'furniture': 'Table', 'consumed_meters': 50, 'max_required_quantity': 4,  'profit': 11}
]

# Derived data
waste = {item['mode']: item['waste'] for item in Table_1_CuttingCombination}
profit_mode = {1: 75, 2: 62, 3: 77, 4: 90, 5: 88, 6: 92}

# Compute net profits for synthetic wood furniture (profit minus processing cost)
net_profit_furniture = {
    item['furniture']: item['profit'] - 0.1 * item['consumed_meters']
    for item in Table_3_SyntheticWoodFurniture
}

# Create Gurobi model
model = gp.Model("Cutting_and_Synthetic_Furniture")

# Decision variables
x = model.addVars([1, 2, 3, 4, 5, 6], vtype=gp.GRB.INTEGER, lb=0, ub=3, name="x")
y_bench = model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=12, name="y_bench")
y_chair = model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=7,  name="y_chair")
y_table = model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=4,  name="y_table")
W = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name="W")

# Objective: maximize total profit (board cutting + net profit from synthetic furniture)
model.setObjective(
    gp.quicksum(profit_mode[p] * x[p] for p in x.keys())
    + net_profit_furniture['Bench'] * y_bench
    + net_profit_furniture['Chair'] * y_chair
    + net_profit_furniture['Table'] * y_table,
    gp.GRB.MAXIMIZE
)

# Constraints

# 1) Factory 2 must process at least one board
model.addConstr(x[4] + x[5] >= 1, name="Factory2Min")

# 2) Mode 6 prerequisite: 3*x6 ≤ sum(x1..x5)
model.addConstr(3 * x[6] <= gp.quicksum(x[p] for p in [1, 2, 3, 4, 5]), name="Mode6Prereq")

# 3) Demand for 25-unit parts
model.addConstr(2 * x[1] + 2 * x[4] + 4 * x[5] >= demand_25, name="Demand25")

# 4) Demand for 40-unit parts
model.addConstr(x[1] + 2 * x[2] + x[3] >= demand_40, name="Demand40")

# 5) Demand for 50-unit parts
model.addConstr(x[3] + x[4] + 2 * x[6] >= demand_50, name="Demand50")

# 6) Total panels used ≤ 20
model.addConstr(gp.quicksum(x[p] for p in x.keys()) <= 20, name="BoardSupply")

# 7) Waste balance for Factory 1
model.addConstr(W == gp.quicksum(waste[p] * x[p] for p in x.keys()), name="WasteBalance")

# 8) Synthetic-wood usage cannot exceed available waste
model.addConstr(W >= 20 * y_bench + 40 * y_chair + 50 * y_table, name="SyntheticUsage")
model.addGenConstrL
# Solve the model
model.optimize()

# Print solution
if model.Status == gp.GRB.OPTIMAL:
    for p in sorted(x.keys()):
        print(f"x_{p} = {x[p].X}")
    print(f"y_bench = {y_bench.X}")
    print(f"y_chair = {y_chair.X}")
    print(f"y_table = {y_table.X}")
    print(f"W = {W.X}")
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found")