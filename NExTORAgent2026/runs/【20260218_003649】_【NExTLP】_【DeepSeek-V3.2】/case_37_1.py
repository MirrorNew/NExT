import gurobipy as gp
from gurobipy import GRB

# Parameters
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
cells = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
magic_sum = 15
A_fixed = 4

# Create model
model = gp.Model("MagicSquare")

# Decision variables: integer values for each cell
x = {}
for cell in cells:
    x[cell] = model.addVar(lb=1, ub=9, vtype=GRB.INTEGER, name=f"x_{cell}")

# Binary variables for all-different constraints
y = {}
for i in range(len(cells)):
    for j in range(i+1, len(cells)):
        y[(cells[i], cells[j])] = model.addVar(vtype=GRB.BINARY, name=f"y_{cells[i]}_{cells[j]}")

# Set objective (feasibility)
model.setObjective(0, GRB.MINIMIZE)

# All-different constraints using indicator constraints
M = 10  # Large enough constant
for i in range(len(cells)):
    for j in range(i+1, len(cells)):
        cell_i = cells[i]
        cell_j = cells[j]
        
        # If y = 1, then x_i >= x_j + 1
        model.addGenConstrIndicator(y[(cell_i, cell_j)], 1, x[cell_i] >= x[cell_j] + 1)
        # If y = 0, then x_j >= x_i + 1
        model.addGenConstrIndicator(y[(cell_i, cell_j)], 0, x[cell_j] >= x[cell_i] + 1)

# Row sum constraints
model.addConstr(x['A'] + x['B'] + x['C'] == magic_sum, "Row1")
model.addConstr(x['D'] + x['E'] + x['F'] == magic_sum, "Row2")
model.addConstr(x['G'] + x['H'] + x['I'] == magic_sum, "Row3")

# Column sum constraints
model.addConstr(x['A'] + x['D'] + x['G'] == magic_sum, "Col1")
model.addConstr(x['B'] + x['E'] + x['H'] == magic_sum, "Col2")
model.addConstr(x['C'] + x['F'] + x['I'] == magic_sum, "Col3")

# Diagonal sum constraints
model.addConstr(x['A'] + x['E'] + x['I'] == magic_sum, "Diag1")
model.addConstr(x['C'] + x['E'] + x['G'] == magic_sum, "Diag2")

# Fixed value constraint for cell A
model.addConstr(x['A'] == A_fixed, "FixedA")

# Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print("Magic Square Solution:")
    print(f"  {x['A'].X:.0f} {x['B'].X:.0f} {x['C'].X:.0f}")
    print(f"  {x['D'].X:.0f} {x['E'].X:.0f} {x['F'].X:.0f}")
    print(f"  {x['G'].X:.0f} {x['H'].X:.0f} {x['I'].X:.0f}")
    
    # Answer the question: If A=4, what is the number in block I?
    print(f"FinalAnswer=【{int(x['I'].X)}】")
else:
    print("No solution found")
    print(f"FinalAnswer=【No feasible solution】")