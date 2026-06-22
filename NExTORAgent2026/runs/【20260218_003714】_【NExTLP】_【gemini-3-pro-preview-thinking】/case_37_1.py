import gurobipy as gp
from gurobipy import GRB

# 1. Import Gurobi and define parameters
Numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
Cells = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
MagicSum = 15
A_param = 4  # Renamed from 'A' to avoid conflict with cell name string 'A'
Table_1_Figure_C_17 = [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']]

# 2. Create the model
model = gp.Model("MagicSquare")

# 3. Create decision variables
# Binary variables: x[c, n] = 1 if cell c contains number n
x = model.addVars(Cells, Numbers, vtype=GRB.BINARY, name="x")

# Integer variables: v[c] = value of cell c (1..9)
v = model.addVars(Cells, vtype=GRB.INTEGER, lb=1, ub=9, name="v")

# 4. Set up the objective function
# This is a constraint satisfaction problem, so we minimize a constant (0)
model.setObjective(0, GRB.MINIMIZE)

# 5. Add Constraints

# Link integer variables v[c] with binary variables x[c, n]
for c in Cells:
    model.addConstr(v[c] == gp.quicksum(n * x[c, n] for n in Numbers), name=f"Link_{c}")

# Constraint: Each cell must contain exactly one number
for c in Cells:
    model.addConstr(gp.quicksum(x[c, n] for n in Numbers) == 1, name=f"Assign_{c}")

# Constraint: All cells must be distinct (AllDifferent) -> Each number used exactly once
for n in Numbers:
    model.addConstr(gp.quicksum(x[c, n] for c in Cells) == 1, name=f"Unique_{n}")

# Constraint: Row Sums
for i in range(3):
    row_cells = Table_1_Figure_C_17[i]
    model.addConstr(gp.quicksum(v[c] for c in row_cells) == MagicSum, name=f"RowSum_{i}")

# Constraint: Column Sums
# Transpose table logic to get columns
for j in range(3):
    col_cells = [Table_1_Figure_C_17[i][j] for i in range(3)]
    model.addConstr(gp.quicksum(v[c] for c in col_cells) == MagicSum, name=f"ColSum_{j}")

# Constraint: Diagonal Sums
# Main diagonal (top-left to bottom-right): (0,0), (1,1), (2,2)
diag1_cells = [Table_1_Figure_C_17[i][i] for i in range(3)]
model.addConstr(gp.quicksum(v[c] for c in diag1_cells) == MagicSum, name="DiagSum_1")

# Anti-diagonal (bottom-left to top-right): (2,0), (1,1), (0,2)
diag2_cells = [Table_1_Figure_C_17[2-i][i] for i in range(3)]
model.addConstr(gp.quicksum(v[c] for c in diag2_cells) == MagicSum, name="DiagSum_2")

# Constraint: Specific value for cell A
model.addConstr(v['A'] == A_param, name="Fix_A")

# 6. Solve the model
model.optimize()

# 7. Print results
if model.status == GRB.OPTIMAL:
    i_value = int(round(v['I'].X))
    print(f"FinalAnswer=【{i_value}】")
else:
    print("FinalAnswer=【No Solution】")