import gurobipy as gp
from gurobipy import GRB

# Define the parameters as specified
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

Table_2_Component = [
    {'length': 25, 'required_quantity': 8, 'profit': 22},
    {'length': 40, 'required_quantity': 6, 'profit': 31},
    {'length': 50, 'required_quantity': 4, 'profit': 46}
]

Table_3_SyntheticWoodFurniture = [
    {'furniture': 'Bench', 'consumed_meters': 20, 'max_required_quantity': 12, 'profit': 3},
    {'furniture': 'Chair', 'consumed_meters': 40, 'max_required_quantity': 7, 'profit': 8},
    {'furniture': 'Table', 'consumed_meters': 50, 'max_required_quantity': 4, 'profit': 11}
]

# Create the model
model = gp.Model("Realm_Optimization")

# --- Decision Variables ---
# x_p: Number of boards cut by mode p (p=1..6)
# Range: 0 <= x_p <= 3 (as per constraint "each cutting mode can only be used for a maximum of three boards")
x = {}
for m in Table_1_CuttingCombination:
    mode_id = m['mode']
    x[mode_id] = model.addVar(lb=0, ub=3, vtype=GRB.INTEGER, name=f"x_{mode_id}")

# y_furniture: Pieces of synthetic-wood furniture produced
# Range defined by max_required_quantity in Table 3
y = {}
for f in Table_3_SyntheticWoodFurniture:
    f_name = f['furniture']
    y[f_name] = model.addVar(lb=0, ub=f['max_required_quantity'], vtype=GRB.INTEGER, name=f"y_{f_name}")

# --- Objective Function ---
# Maximize Total Profit = Profit from cut parts + Profit from synthetic furniture
# Note: Synthetic furniture profit needs to account for processing cost (0.1 per meter)
# The context provided simplified coefficients: 1y_bench, 4y_chair, 6y_table
# Let's verify: Bench: 3 - 20*0.1 = 1. Chair: 8 - 40*0.1 = 4. Table: 11 - 50*0.1 = 6.
# Component profit for each mode:
# Mode 1 (40, 25, 25): 31 + 22 + 22 = 75
# Mode 2 (40, 40): 31 + 31 = 62
# Mode 3 (50, 40): 46 + 31 = 77
# Mode 4 (50, 25, 25): 46 + 22 + 22 = 90
# Mode 5 (25, 25, 25, 25): 4*22 = 88
# Mode 6 (50, 50): 2*46 = 92

# Pre-calculate mode profits
comp_profit_map = {item['length']: item['profit'] for item in Table_2_Component}
mode_profits = {}
for m in Table_1_CuttingCombination:
    m_id = m['mode']
    parts = m['cutting_combination']
    p_val = sum(comp_profit_map[l] for l in parts)
    mode_profits[m_id] = p_val

# Pre-calculate furniture net profits
furn_profits = {}
for f in Table_3_SyntheticWoodFurniture:
    cost = f['consumed_meters'] * 0.1
    net = f['profit'] - cost
    furn_profits[f['furniture']] = net

obj_expr = gp.quicksum(mode_profits[p] * x[p] for p in x) + \
           gp.quicksum(furn_profits[f] * y[f] for f in y)

model.setObjective(obj_expr, GRB.MAXIMIZE)

# --- Constraints ---

# 1. Board supply limit
model.addConstr(gp.quicksum(x[p] for p in x) <= 20, "BoardSupplyLimit")

# 2. Factory 2 minimum processing (Modes 4 and 5)
model.addConstr(x[4] + x[5] >= 1, "Factory2MinProcessing")

# 3. Prerequisite for mode 6
# "any factory must process any mode of wood three times before it can cut mode 6"
# Context Interpretation: x_6 <= (x_1+x_2+x_3)/3 + (x_4+x_5)/3
# Equivalent to: 3 * x_6 <= sum(x_1 to x_5)
model.addConstr(3 * x[6] <= x[1] + x[2] + x[3] + x[4] + x[5], "Mode6Prerequisite")

# 4. Demand satisfaction
# Helper to count parts of length L produced by mode p
def count_parts(mode_idx, length):
    comb = next(item['cutting_combination'] for item in Table_1_CuttingCombination if item['mode'] == mode_idx)
    return comb.count(length)

# Demand 25
produced_25 = gp.quicksum(count_parts(p, 25) * x[p] for p in x)
model.addConstr(produced_25 >= demand_25, "Demand_25")

# Demand 40
produced_40 = gp.quicksum(count_parts(p, 40) * x[p] for p in x)
model.addConstr(produced_40 >= demand_40, "Demand_40")

# Demand 50
produced_50 = gp.quicksum(count_parts(p, 50) * x[p] for p in x)
model.addConstr(produced_50 >= demand_50, "Demand_50")

# 5. Waste Balance
# Waste is generated only by Factory 1 (Modes 1, 2, 3) as per problem text, 
# though Table 1 shows 0 waste for 4, 5, 6 anyway.
total_waste = gp.quicksum(next(item['waste'] for item in Table_1_CuttingCombination if item['mode'] == p) * x[p] for p in x)

consumed_waste = gp.quicksum(next(item['consumed_meters'] for item in Table_3_SyntheticWoodFurniture if item['furniture'] == f) * y[f] for f in y)

model.addConstr(consumed_waste <= total_waste, "WasteUsage")

# Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print(f"Optimal Objective Value: {model.objVal}")
    for v in model.getVars():
        if v.x > 0:
            print(f"{v.varName}: {v.x}")
            
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found.")