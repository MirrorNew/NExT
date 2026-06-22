import gurobipy as gp

# Parameters
standard_panel_length = 100
factory1_modes = [1, 2, 3]
factory2_modes = [4, 5]
demand_25 = 8
demand_40 = 6
demand_50 = 4

# Cutting combination data from Table 1
cutting_data = [
    {'mode': 1, 'cutting_combination': [40, 25, 25], 'waste': 10},
    {'mode': 2, 'cutting_combination': [40, 40], 'waste': 20},
    {'mode': 3, 'cutting_combination': [50, 40], 'waste': 10},
    {'mode': 4, 'cutting_combination': [50, 25, 25], 'waste': 0},
    {'mode': 5, 'cutting_combination': [25, 25, 25, 25], 'waste': 0},
    {'mode': 6, 'cutting_combination': [50, 50], 'waste': 0}
]

# Component data from Table 2
component_data = [
    {'length': 25, 'required_quantity': 8, 'profit': 22},
    {'length': 40, 'required_quantity': 6, 'profit': 31},
    {'length': 50, 'required_quantity': 4, 'profit': 46}
]

# Synthetic wood furniture data from Table 3
furniture_data = [
    {'furniture': 'Bench', 'consumed_meters': 20, 'max_required_quantity': 12, 'profit': 3},
    {'furniture': 'Chair', 'consumed_meters': 40, 'max_required_quantity': 7, 'profit': 8},
    {'furniture': 'Table', 'consumed_meters': 50, 'max_required_quantity': 4, 'profit': 11}
]

# Create model
model = gp.Model("PanelCuttingOptimization")

# Decision variables
x = {}
for p in range(1, 7):
    x[p] = model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=3, name=f"x_{p}")

y_bench = model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=12, name="y_bench")
y_chair = model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=7, name="y_chair")
y_table = model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=4, name="y_table")
W = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name="W")

# Set objective
# Profit coefficients from objective expression: 75x1 + 62x2 + 77x3 + 90x4 + 88x5 + 92x6 + 1y_bench + 4y_chair + 6y_table
objective = (75*x[1] + 62*x[2] + 77*x[3] + 90*x[4] + 88*x[5] + 92*x[6] + 
             1*y_bench + 4*y_chair + 6*y_table)
model.setObjective(objective, gp.GRB.MAXIMIZE)

# Constraints
# 1. Factory 2 minimum processing: x4 + x5 ≥ 1
model.addConstr(x[4] + x[5] >= 1, "factory2_min")

# 2. Prerequisite for mode 6: x6 ≤ (x1+x2+x3)/3 + (x4+x5)/3
# Multiply both sides by 3 to avoid division: 3*x6 ≤ (x1+x2+x3) + (x4+x5)
model.addConstr(3*x[6] <= x[1] + x[2] + x[3] + x[4] + x[5], "mode6_prerequisite")

# 3. Demand for 25-unit parts: 2x1 + 2x4 + 4x5 ≥ 8
model.addConstr(2*x[1] + 2*x[4] + 4*x[5] >= demand_25, "demand_25")

# 4. Demand for 40-unit parts: x1 + 2x2 + x3 ≥ 6
model.addConstr(x[1] + 2*x[2] + x[3] >= demand_40, "demand_40")

# 5. Demand for 50-unit parts: x3 + x4 + 2x6 ≥ 4
model.addConstr(x[3] + x[4] + 2*x[6] >= demand_50, "demand_50")

# 6. Board supply limit: sum(x_p) ≤ 20
model.addConstr(gp.quicksum(x[p] for p in range(1, 7)) <= 20, "board_supply_limit")

# 7. Waste balance (Factory 1): W = 10x1 + 20x2 + 10x3
model.addConstr(W == 10*x[1] + 20*x[2] + 10*x[3], "waste_balance")

# 8. Synthetic-wood usage: W ≥ 20y_bench + 40y_chair + 50y_table
model.addConstr(W >= 20*y_bench + 40*y_chair + 50*y_table, "synthetic_wood_usage")

# Solve the model
model.optimize()

# Print results
if model.status == gp.GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"Objective value (total profit): {model.objVal}")
    print("\nCutting mode usage:")
    for p in range(1, 7):
        print(f"  Mode {p}: {x[p].X}")
    
    print("\nSynthetic wood furniture production:")
    print(f"  Benches: {y_bench.X}")
    print(f"  Chairs: {y_chair.X}")
    print(f"  Tables: {y_table.X}")
    
    print(f"\nTotal usable waste (meters): {W.X}")
    
    # Calculate total boards used
    total_boards = sum(x[p].X for p in range(1, 7))
    print(f"Total boards used: {total_boards}")
    
    # Output the final answer as required
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found.")
    print(f"FinalAnswer=【0】")