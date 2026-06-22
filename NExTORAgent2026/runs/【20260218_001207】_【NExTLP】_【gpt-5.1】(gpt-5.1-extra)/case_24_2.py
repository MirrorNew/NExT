import gurobipy as gp
from gurobipy import GRB

# -------------------------------------------------
# 1. Parameters (strictly from Parameters List)
# -------------------------------------------------
standard_board_length_unit = 100
num_factories = 2
factory_1_modes = [1, 2, 3]
factory_2_modes = [4, 5]
min_processing_factory_2 = 1
min_processing_any_mode_before_mode_6 = 3

component_lengths = [25, 40, 50]
component_required_quantities = [8, 6, 4]

per_mode_max_boards = 3

Table_1_modes = [1, 2, 3, 4, 5, 6]
Table_1_cutting_combinations = [
    [40, 25, 25],          # mode 1
    [40, 40],              # mode 2
    [50, 40],              # mode 3
    [50, 25, 25],          # mode 4
    [25, 25, 25, 25],      # mode 5
    [50, 50],              # mode 6
]
Table_1_waste_unit = [10, 20, 10, 0, 0, 0]

Table_2_component_lengths = [25, 40, 50]
Table_2_required_quantities = [8, 6, 4]
Table_2_profit_yuan = [22, 31, 46]

synthetic_wood_processing_cost_per_meter = 0.1

synthetic_furniture_types = ["Bench", "Chair", "Table"]
Table_3_consumed_boards_meters = [20, 40, 50]
Table_3_max_required_quantity_pieces = [12, 7, 4]
Table_3_profit_yuan = [3, 8, 11]

shipment_num_standard_boards = 20

# Profit aliases (for convenience, still from Parameters List)
profit_25 = Table_2_profit_yuan[0]
profit_40 = Table_2_profit_yuan[1]
profit_50 = Table_2_profit_yuan[2]

furniture_profit_bench = Table_3_profit_yuan[0]
furniture_profit_chair = Table_3_profit_yuan[1]
furniture_profit_table = Table_3_profit_yuan[2]

# -------------------------------------------------
# 2. Create Gurobi model
# -------------------------------------------------
model = gp.Model("Panel_Cutting_and_Synthetic_Wood_Profit_Maximization")

# -------------------------------------------------
# 3. Decision variables
# -------------------------------------------------

# Cutting mode variables x_1,...,x_6 (integer, 0..3)
x = {}
for p in Table_1_modes:
    x[p] = model.addVar(
        vtype=GRB.INTEGER,
        lb=0,
        ub=per_mode_max_boards,
        name=f"x_{p}",
    )

# Binary activation variable for mode 6
y6 = model.addVar(vtype=GRB.BINARY, name="y_6")

# Synthetic furniture variables: b, c, t (bench, chair, table)
b = model.addVar(
    vtype=GRB.INTEGER,
    lb=0,
    ub=Table_3_max_required_quantity_pieces[0],
    name="b",
)
c = model.addVar(
    vtype=GRB.INTEGER,
    lb=0,
    ub=Table_3_max_required_quantity_pieces[1],
    name="c",
)
t = model.addVar(
    vtype=GRB.INTEGER,
    lb=0,
    ub=Table_3_max_required_quantity_pieces[2],
    name="t",
)

# Synthetic-wood availability W (continuous, >=0)
W = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="W")

model.update()

# -------------------------------------------------
# 4. Constraints
# -------------------------------------------------

# Non-negativity and upper bounds of x_p, b, c, t are already enforced by variable bounds.
# Binary nature of y6 is enforced by its vtype.

# Factory 2 must process at least one board: x_4 + x_5 >= 1
model.addConstr(
    x[4] + x[5] >= min_processing_factory_2,
    name="factory_2_min_processing",
)

# Mode-6 activation link via indicator (no big-M):
# If y6 == 0 then x6 == 0; if y6 == 1 then x6 <= 3
model.addGenConstrIndicator(
    y6, 0, x[6] == 0, name="ind_y6_0_x6_0"
)
model.addGenConstrIndicator(
    y6, 1, x[6] <= per_mode_max_boards, name="ind_y6_1_x6_le_3"
)

# Precedence for mode 6: any factory must process any mode of wood three times
# before using mode 6. We interpret this as: total boards in modes 1–5 at
# least 3 if mode 6 is used.
model.addGenConstrIndicator(
    y6,
    1,
    x[1] + x[2] + x[3] + x[4] + x[5] >= min_processing_any_mode_before_mode_6,
    name="ind_y6_1_precedence_3_boards"
)

# Board availability limit: sum_p x_p <= 20
model.addConstr(
    gp.quicksum(x[p] for p in Table_1_modes) <= shipment_num_standard_boards,
    name="board_availability",
)

# Demand satisfaction – 25-unit parts: 2x1 + 2x4 + 4x5 >= 8
model.addConstr(
    2 * x[1] + 2 * x[4] + 4 * x[5] >= Table_2_required_quantities[0],
    name="demand_25",
)

# Demand satisfaction – 40-unit parts: x1 + 2x2 + x3 >= 6
model.addConstr(
    x[1] + 2 * x[2] + x[3] >= Table_2_required_quantities[1],
    name="demand_40",
)

# Demand satisfaction – 50-unit parts: x3 + x4 + 2x6 >= 4
model.addConstr(
    x[3] + x[4] + 2 * x[6] >= Table_2_required_quantities[2],
    name="demand_50",
)

# Synthetic wood availability definition (Factory 1): W = 10x1 + 20x2 + 10x3
model.addConstr(
    W == 10 * x[1] + 20 * x[2] + 10 * x[3],
    name="synthetic_wood_definition",
)

# Synthetic furniture resource constraint: 20b + 40c + 50t <= W
model.addConstr(
    20 * b + 40 * c + 50 * t <= W,
    name="synthetic_furniture_resource",
)

# Upper bounds of synthetic furniture variables are already encoded in variable defs.

# -------------------------------------------------
# 5. Objective function: profit maximization
# -------------------------------------------------
# Π_total =
# 22(2x1 + 2x4 + 4x5) + 31(x1 + 2x2 + x3)
# + 46(x3 + x4 + 2x6)
# + 3b + 8c + 11t - 0.1(20b + 40c + 50t)

component_profit_expr = (
    profit_25 * (2 * x[1] + 2 * x[4] + 4 * x[5])
    + profit_40 * (x[1] + 2 * x[2] + x[3])
    + profit_50 * (x[3] + x[4] + 2 * x[6])
)

furniture_revenue_expr = (
    furniture_profit_bench * b
    + furniture_profit_chair * c
    + furniture_profit_table * t
)

processing_cost_expr = synthetic_wood_processing_cost_per_meter * (
    20 * b + 40 * c + 50 * t
)

total_profit_expr = component_profit_expr + furniture_revenue_expr - processing_cost_expr

model.setObjective(total_profit_expr, GRB.MAXIMIZE)

# -------------------------------------------------
# 6. Solve model
# -------------------------------------------------
model.optimize()

# -------------------------------------------------
# 7. Print results and FinalAnswer
# -------------------------------------------------
if model.status == GRB.OPTIMAL:
    total_profit_value = model.objVal
    print("\nOptimal solution found.")
    print(f"Total profit: {total_profit_value:.2f} yuan\n")

    print("Cutting plan (boards per mode):")
    for p in Table_1_modes:
        print(f"  x_{p} (mode {p}): {int(round(x[p].X))}")

    print(f"\nMode 6 activated (y_6): {int(round(y6.X))}")

    print("\nSynthetic wood and furniture:")
    print(f"  W (synthetic wood meters): {W.X:.2f}")
    print(f"  b (benches): {int(round(b.X))}")
    print(f"  c (chairs): {int(round(c.X))}")
    print(f"  t (tables): {int(round(t.X))}")

    # Final answer as required
    print(f"FinalAnswer=【{total_profit_value:.2f}】")
else:
    print(f"Optimization ended with status {model.status}. No optimal solution available.")
    # If no optimal solution, still output something for FinalAnswer
    print("FinalAnswer=【NaN】")