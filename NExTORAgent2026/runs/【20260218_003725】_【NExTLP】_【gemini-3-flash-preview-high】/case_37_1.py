import gurobipy as gp

# 1. Define all parameter matrices and data inputs.
# These parameters are strictly derived from the provided list.
Numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
Cells = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
MagicSum = 15
A_fixed_val = 4
Table_1_Figure_C_17 = [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']]

# 2. Create the model.
model = gp.Model("MagicSquare")

# 3. Create decision variables.
# We use integer variables for each cell, bounded between 1 and 9.
cell_vars = model.addVars(Cells, lb=1, ub=9, vtype=gp.GRB.INTEGER, name="cell")

# 4. Set up the objective function.
# This is a constraint satisfaction problem, so we minimize a constant (0).
model.setObjective(0, gp.GRB.MINIMIZE)

# 5. Add all constraints.
# Uniqueness: All cells must contain pairwise distinct values.
model.addGenConstrAllDiff([cell_vars[c] for c in Cells], name="UniqueValues")

# Row constraints: The sum of numbers in each row must be equal to MagicSum (15).
for row in Table_1_Figure_C_17:
    model.addConstr(gp.quicksum(cell_vars[c] for c in row) == MagicSum, name=f"RowSum_{row}")

# Column constraints: The sum of numbers in each column must be equal to MagicSum (15).
for j in range(3):
    model.addConstr(gp.quicksum(cell_vars[Table_1_Figure_C_17[i][j]] for i in range(3)) == MagicSum, name=f"ColSum_{j}")

# Diagonal constraints: The sum of the two main diagonals must be equal to MagicSum (15).
# Diagonal 1: top-left (A) to bottom-right (I)
model.addConstr(cell_vars['A'] + cell_vars['E'] + cell_vars['I'] == MagicSum, name="Diag1Sum")
# Diagonal 2: top-right (C) to bottom-left (G)
model.addConstr(cell_vars['C'] + cell_vars['E'] + cell_vars['G'] == MagicSum, name="Diag2Sum")

# Block A Value constraint: The value in block A is given as 4.
model.addConstr(cell_vars['A'] == A_fixed_val, name="BlockAValue")

# 6. Solve the model.
model.optimize()

# 7. Print results and the final answer to the specific question.
if model.status == gp.GRB.OPTIMAL:
    # Retrieve the value assigned to block I
    ans_I = int(cell_vars['I'].X)
    print(f"FinalAnswer=【{ans_I}】")
else:
    print("No feasible solution found.")