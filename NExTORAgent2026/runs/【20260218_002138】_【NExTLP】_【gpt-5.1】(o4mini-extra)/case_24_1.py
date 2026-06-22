import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters (from Parameters List)
# =========================

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
    {'mode': 6, 'cutting_combination': [50, 50], 'waste': 0},
]

Table_2_Component = [
    {'length': 25, 'required_quantity': 8, 'profit': 22},
    {'length': 40, 'required_quantity': 6, 'profit': 31},
    {'length': 50, 'required_quantity': 4, 'profit': 46},
]

Table_3_SyntheticWoodFurniture = [
    {'furniture': 'Bench', 'consumed_meters': 20, 'max_required_quantity': 12, 'profit': 3},
    {'furniture': 'Chair', 'consumed_meters': 40, 'max_required_quantity': 7, 'profit': 8},
    {'furniture': 'Table', 'consumed_meters': 50, 'max_required_quantity': 4, 'profit': 11},
]

# Mode profits and furniture profits given by the validated model
mode_profits = {1: 75, 2: 62, 3: 77, 4: 90, 5: 88, 6: 92}
bench_profit = 1
chair_profit = 4
table_profit = 6

# Waste per mode (only factory 1 modes have nonzero waste usable as synthetic wood)
waste_per_mode = {row['mode']: row['waste'] for row in Table_1_CuttingCombination}

# Synthetic wood furniture parameters
furniture_consumption = {
    'Bench': Table_3_SyntheticWoodFurniture[0]['consumed_meters'],
    'Chair': Table_3_SyntheticWoodFurniture[1]['consumed_meters'],
    'Table': Table_3_SyntheticWoodFurniture[2]['consumed_meters'],
}
furniture_max_demand = {
    'Bench': Table_3_SyntheticWoodFurniture[0]['max_required_quantity'],
    'Chair': Table_3_SyntheticWoodFurniture[1]['max_required_quantity'],
    'Table': Table_3_SyntheticWoodFurniture[2]['max_required_quantity'],
}

# =========================
# 2. Create model
# =========================

model = gp.Model("Cutting_and_SyntheticWood_Profit_Maximization")

# =========================
# 3. Decision Variables
# =========================

# x_p: number of boards cut by mode p (integer, 0..3)
modes = [1, 2, 3, 4, 5, 6]
x = model.addVars(modes, vtype=GRB.INTEGER, lb=0, ub=3, name="x")

# Synthetic wood furniture quantities (integer)
y_bench = model.addVar(vtype=GRB.INTEGER, lb=0, ub=furniture_max_demand['Bench'], name="y_bench")
y_chair = model.addVar(vtype=GRB.INTEGER, lb=0, ub=furniture_max_demand['Chair'], name="y_chair")
y_table = model.addVar(vtype=GRB.INTEGER, lb=0, ub=furniture_max_demand['Table'], name="y_table")

# Total waste from Factory 1 usable for synthetic wood (continuous)
W = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="W")

# =========================
# 4. Objective function
# =========================

model.setObjective(
    mode_profits[1] * x[1]
    + mode_profits[2] * x[2]
    + mode_profits[3] * x[3]
    + mode_profits[4] * x[4]
    + mode_profits[5] * x[5]
    + mode_profits[6] * x[6]
    + bench_profit * y_bench
    + chair_profit * y_chair
    + table_profit * y_table,
    GRB.MAXIMIZE
)

# =========================
# 5. Constraints
# =========================

# 5.1 Non-negativity and upper bounds for x already enforced via lb, ub

# 5.2 Factory 2 minimum processing: x4 + x5 >= 1
model.addConstr(x[4] + x[5] >= 1, name="Factory2_min_processing")

# 5.3 Prerequisite for mode 6:
#     x6 <= (x1 + x2 + x3)/3 + (x4 + x5)/3
model.addConstr(
    x[6] <= (x[1] + x[2] + x[3]) / 3.0 + (x[4] + x[5]) / 3.0,
    name="Mode6_prerequisite"
)

# 5.4 Demand satisfaction for parts

# 25-unit parts: 2x1 + 2x4 + 4x5 >= 8
model.addConstr(2 * x[1] + 2 * x[4] + 4 * x[5] >= demand_25, name="Demand_25")

# 40-unit parts: x1 + 2x2 + x3 >= 6
model.addConstr(x[1] + 2 * x[2] + x[3] >= demand_40, name="Demand_40")

# 50-unit parts: x3 + x4 + 2x6 >= 4
model.addConstr(x[3] + x[4] + 2 * x[6] >= demand_50, name="Demand_50")

# 5.5 Board supply limit: sum_{p=1..6} x_p <= 20
model.addConstr(gp.quicksum(x[p] for p in modes) <= 20, name="Board_supply_limit")

# 5.6 Waste balance (Factory 1): W = 10x1 + 20x2 + 10x3
model.addConstr(
    W == waste_per_mode[1] * x[1]
    + waste_per_mode[2] * x[2]
    + waste_per_mode[3] * x[3],
    name="Waste_balance"
)

# 5.7 Synthetic wood usage: W >= 20 y_bench + 40 y_chair + 50 y_table
model.addConstr(
    W >= furniture_consumption['Bench'] * y_bench
    + furniture_consumption['Chair'] * y_chair
    + furniture_consumption['Table'] * y_table,
    name="Synthetic_wood_usage"
)

# 5.8 Max synthetic-wood furniture demand already enforced via variable upper bounds

# =========================
# 6. Optimize
# =========================

model.optimize()

# =========================
# 7. Print results
# =========================

if model.SolCount > 0:
    total_profit = model.ObjVal

    print("Optimal solution found.")
    print(f"Total profit: {total_profit}")

    for p in modes:
        print(f"x_{p} (boards in mode {p}) = {x[p].X}")

    print(f"y_bench = {y_bench.X}")
    print(f"y_chair = {y_chair.X}")
    print(f"y_table = {y_table.X}")
    print(f"W (waste usable for synthetic wood) = {W.X}")

    # Final answer required by the problem statement: total profit
    print(f"FinalAnswer=【{total_profit}】")
else:
    print("No feasible solution found.")
    # In case of infeasibility, still print a final answer placeholder
    print("FinalAnswer=【NaN】")