import gurobipy as gp

# 1. Define all parameter matrices and data inputs.
# These parameters are derived from the problem description and the provided list.
Numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
Cells = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
MagicSum = 15
A_val = 4
# Table_1_Figure_C_17 represents the layout of the 3x3 square as seen in Figure C-17.
Table_1_Figure_C_17 = [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']]

# 2. Create the model.
model = gp.Model("MagicSquare")

# 3. Create decision variables.
# cell_vars: Integer variables representing the value in each cell (1-9).
cell_vars = model.addVars(Cells, lb=1, ub=9, vtype=gp.GRB.INTEGER, name="cell")
# x: Binary variables for assignment (x[c, v] = 1 if cell c contains number v).
x = model.addVars(Cells, Numbers, vtype=gp.GRB.BINARY, name="assign")

# 4. Set up the objective function.
# This is a feasibility problem, so we minimize 0.
model.setObjective(0, gp.GRB.MINIMIZE)

# 5. Add all constraints.
# Each cell must contain exactly one number.
for c in Cells:
    model.addConstr(gp.quicksum(x[c, v] for v in Numbers) == 1, name=f"OneValuePerCell_{c}")
    # ATTENTION 2: Link integer and binary variables using addGenConstrIndicator.
    for v in Numbers:
        # If binary variable x[c, v] is 1, then the cell_vars[c] must equal v.
        model.addGenConstrIndicator(x[c, v], 1, cell_vars[c] == v)

# Each number must be used exactly once in the grid.
for v in Numbers:
    model.addConstr(gp.quicksum(x[c, v] for c in Cells) == 1, name=f"OneCellPerValue_{v}")

# Row Sums: The sum of numbers in each row must be 15.
for i, row_cells in enumerate(Table_1_Figure_C_17):
    model.addConstr(gp.quicksum(cell_vars[c] for c in row_cells) == MagicSum, name=f"RowSum_{i}")

# Column Sums: The sum of numbers in each column must be 15.
for j in range(3):
    model.addConstr(gp.quicksum(cell_vars[Table_1_Figure_C_17[i][j]] for i in range(3)) == MagicSum, name=f"ColSum_{j}")

# Diagonal Sums: The sum of numbers on the two main diagonals must be 15.
# Diagonal 1: A-E-I
model.addConstr(cell_vars['A'] + cell_vars['E'] + cell_vars['I'] == MagicSum, name="Diag1Sum")
# Diagonal 2: C-E-G
model.addConstr(cell_vars['C'] + cell_vars['E'] + cell_vars['G'] == MagicSum, name="Diag2Sum")

# Specific constraint: Block A must be the value 4.
model.addConstr(cell_vars['A'] == A_val, name="FixValueA")

# 6. Solve the model.
model.optimize()

# 7. Print results and the final answer to the specific question.
if model.status == gp.GRB.OPTIMAL:
    # Retrieve the value of I from the solved model.
    ans_I = int(cell_vars['I'].X + 0.5)
    print(f"FinalAnswer=【{ans_I}】")
else:
    print("No feasible solution found.")