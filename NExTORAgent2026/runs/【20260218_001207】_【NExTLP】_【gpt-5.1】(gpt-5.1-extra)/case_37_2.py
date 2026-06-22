import gurobipy as gp

# Solve the 3x3 magic square with A fixed to 4, and output the value in cell I
# FinalAnswer should be the value in cell I.

# 1. Define all parameter matrices and data inputs (use only the provided Parameters List)
magic_square_size = 3
magic_square_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
magic_constant = 15
cell_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
fixed_value_A = 4
Table_1_layout = [['A', 'B', 'C'],
                  ['D', 'E', 'F'],
                  ['G', 'H', 'I']]

# 2. Create Gurobi model
model = gp.Model("MagicSquare3x3")

# 3. Create decision variables
# Binary variable x[p, v] = 1 if cell p takes value v, 0 otherwise
x = model.addVars(
    cell_labels,
    magic_square_numbers,
    vtype=gp.GRB.BINARY,
    name="x"
)

# 4. Set up the objective function: Minimize 0 (pure feasibility problem)
model.setObjective(0, gp.GRB.MINIMIZE)

# 5. Add all constraints

# Each position takes exactly one value
for p in cell_labels:
    model.addConstr(
        gp.quicksum(x[p, v] for v in magic_square_numbers) == 1,
        name=f"one_value_{p}"
    )

# Each value is used exactly once
for v in magic_square_numbers:
    model.addConstr(
        gp.quicksum(x[p, v] for p in cell_labels) == 1,
        name=f"use_value_{v}"
    )

# Magic sum constraints for rows
# Row 1: A + B + C = magic_constant
model.addConstr(
    gp.quicksum(v * x['A', v] for v in magic_square_numbers) +
    gp.quicksum(v * x['B', v] for v in magic_square_numbers) +
    gp.quicksum(v * x['C', v] for v in magic_square_numbers) ==
    magic_constant,
    name="row1"
)

# Row 2: D + E + F = magic_constant
model.addConstr(
    gp.quicksum(v * x['D', v] for v in magic_square_numbers) +
    gp.quicksum(v * x['E', v] for v in magic_square_numbers) +
    gp.quicksum(v * x['F', v] for v in magic_square_numbers) ==
    magic_constant,
    name="row2"
)

# Row 3: G + H + I = magic_constant
model.addConstr(
    gp.quicksum(v * x['G', v] for v in magic_square_numbers) +
    gp.quicksum(v * x['H', v] for v in magic_square_numbers) +
    gp.quicksum(v * x['I', v] for v in magic_square_numbers) ==
    magic_constant,
    name="row3"
)

# Magic sum constraints for columns
# Column 1: A + D + G = magic_constant
model.addConstr(
    gp.quicksum(v * x['A', v] for v in magic_square_numbers) +
    gp.quicksum(v * x['D', v] for v in magic_square_numbers) +
    gp.quicksum(v * x['G', v] for v in magic_square_numbers) ==
    magic_constant,
    name="col1"
)

# Column 2: B + E + H = magic_constant
model.addConstr(
    gp.quicksum(v * x['B', v] for v in magic_square_numbers) +
    gp.quicksum(v * x['E', v] for v in magic_square_numbers) +
    gp.quicksum(v * x['H', v] for v in magic_square_numbers) ==
    magic_constant,
    name="col2"
)

# Column 3: C + F + I = magic_constant
model.addConstr(
    gp.quicksum(v * x['C', v] for v in magic_square_numbers) +
    gp.quicksum(v * x['F', v] for v in magic_square_numbers) +
    gp.quicksum(v * x['I', v] for v in magic_square_numbers) ==
    magic_constant,
    name="col3"
)

# Magic sum constraints for diagonals
# Main diagonal: A + E + I = magic_constant
model.addConstr(
    gp.quicksum(v * x['A', v] for v in magic_square_numbers) +
    gp.quicksum(v * x['E', v] for v in magic_square_numbers) +
    gp.quicksum(v * x['I', v] for v in magic_square_numbers) ==
    magic_constant,
    name="diag_main"
)

# Other diagonal: C + E + G = magic_constant
model.addConstr(
    gp.quicksum(v * x['C', v] for v in magic_square_numbers) +
    gp.quicksum(v * x['E', v] for v in magic_square_numbers) +
    gp.quicksum(v * x['G', v] for v in magic_square_numbers) ==
    magic_constant,
    name="diag_other"
)

# Fixed value of A: A = fixed_value_A
model.addConstr(x['A', fixed_value_A] == 1, name="fix_A")

# 6. Solve the model
model.optimize()

# 7. Print results and the required FinalAnswer
if model.status == gp.GRB.OPTIMAL:
    # Extract the chosen value for each cell
    solution = {}
    for p in cell_labels:
        for v in magic_square_numbers:
            if x[p, v].X > 0.5:
                solution[p] = v
                break

    # Print magic square in the specified layout
    print("Optimal magic square:")
    for row in Table_1_layout:
        print(" ".join(str(solution[cell]) for cell in row))

    # Value in block I (this is the answer to the question)
    value_I = solution['I']
    print(f"Number in block A is fixed to {fixed_value_A}.")
    print(f"Number in block I is: {value_I}")

    # Final required output format
    print(f"FinalAnswer=【{value_I}】")
else:
    print(f"No optimal solution found. Status code: {model.status}")
    # If no solution, still print FinalAnswer with a placeholder (e.g., -1)
    print("FinalAnswer=【-1】")