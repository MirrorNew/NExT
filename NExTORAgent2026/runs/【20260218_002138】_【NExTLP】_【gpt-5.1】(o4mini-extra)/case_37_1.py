import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters and data
# =========================

Numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
Cells = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
MagicSum = 15
A_fixed_value = 4
Table_1_Figure_C_17 = [['A', 'B', 'C'],
                       ['D', 'E', 'F'],
                       ['G', 'H', 'I']]

# =========================
# 2. Create model
# =========================

model = gp.Model("MagicSquare_3x3")

# =========================
# 3. Decision variables
# =========================
# x[c, v] = 1 if cell c takes value v, 0 otherwise
x = model.addVars(Cells, Numbers, vtype=GRB.BINARY, name="x")

# =========================
# 4. Objective function
# =========================
# Pure feasibility problem: minimize 0
model.setObjective(0, GRB.MINIMIZE)

# =========================
# 5. Constraints
# =========================

# Each cell takes exactly one value
for c in Cells:
    model.addConstr(gp.quicksum(x[c, v] for v in Numbers) == 1,
                    name=f"Cell_{c}_OneValue")

# Each number 1..9 is used exactly once (all-different)
for v in Numbers:
    model.addConstr(gp.quicksum(x[c, v] for c in Cells) == 1,
                    name=f"Number_{v}_UsedOnce")

# Helper: expression for numeric value of a cell
def cell_value(cell):
    return gp.quicksum(v * x[cell, v] for v in Numbers)

# Row sums = MagicSum
# Table_1_Figure_C_17 gives the row structure
for r_index, row in enumerate(Table_1_Figure_C_17):
    model.addConstr(gp.quicksum(cell_value(cell) for cell in row) == MagicSum,
                    name=f"Row_{r_index+1}_Sum")

# Column sums = MagicSum
# Derive columns from Table_1_Figure_C_17
num_rows = len(Table_1_Figure_C_17)
num_cols = len(Table_1_Figure_C_17[0])

for j in range(num_cols):
    col_cells = [Table_1_Figure_C_17[i][j] for i in range(num_rows)]
    model.addConstr(gp.quicksum(cell_value(cell) for cell in col_cells) == MagicSum,
                    name=f"Col_{j+1}_Sum")

# Diagonal sums = MagicSum
# Main diagonal: (0,0), (1,1), (2,2)
diag1_cells = [Table_1_Figure_C_17[i][i] for i in range(num_rows)]
model.addConstr(gp.quicksum(cell_value(cell) for cell in diag1_cells) == MagicSum,
                name="Diag1_Sum")

# Other diagonal: (0,2), (1,1), (2,0)
diag2_cells = [Table_1_Figure_C_17[i][num_cols - 1 - i] for i in range(num_rows)]
model.addConstr(gp.quicksum(cell_value(cell) for cell in diag2_cells) == MagicSum,
                name="Diag2_Sum")

# Fix A = 4
model.addConstr(x['A', A_fixed_value] == 1, name="Fix_A_4")

# =========================
# 6. Optimize
# =========================

model.optimize()

# =========================
# 7. Print results
# =========================

if model.status == GRB.OPTIMAL:
    # Reconstruct values for all cells
    cell_assignments = {}
    for c in Cells:
        for v in Numbers:
            if x[c, v].X > 0.5:
                cell_assignments[c] = v
                break

    # Print full magic square
    print("Optimal 3x3 magic square (A..I):")
    for row in Table_1_Figure_C_17:
        print([cell_assignments[cell] for cell in row])

    # Extract the answer: value in block I
    I_value = cell_assignments['I']
    print(f"The value in block I is: {I_value}")

    # Final answer line (required format)
    print(f"FinalAnswer=【{I_value}】")
else:
    print("No optimal solution found.")
    # If no solution, still print a FinalAnswer with a placeholder (e.g., -1)
    print("FinalAnswer=【-1】")