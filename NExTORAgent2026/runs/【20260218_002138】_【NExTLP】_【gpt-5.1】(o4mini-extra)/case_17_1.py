import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters and Data
# =========================

# From Parameters List (must use exactly these values)

number_of_assembly_lines = 3
assembly_line_names = ['A', 'B', 'C']
tasks = ['A', 'B', 'C']
task_product_map = {'A': 'beef', 'B': 'cod', 'C': 'shrimp'}
process_sequence = ['initial cutting', 'deep freezing', 'packaging']
ice_coating_rate_range = (0.06, 0.08)
process_line_map = {
    'initial cutting': 'A',
    'deep freezing': 'B',
    'packaging': 'C'
}
precedence_constraints = [['initial cutting', 'deep freezing'],
                          ['deep freezing', 'packaging']]
transfer_time = 0.5
transfer_must_be_within = (6, 20)
no_transfer_after_packaging = True
parallel_start_allowed = True
capacity_line_A = 1
max_concurrent_batches = 2
capacity_line_B = 1
capacity_line_C = 1
shift_start = 6
shift_end = 20
daily_operation_hours = 14
time_origin = 0
time_unit = ['hours']
max_construction_days = 3
Table_1_ProcessingTimes = [
    ['A beef', 3, 2, 1],
    ['B cod', 2, 4, 1],
    ['C Shrimp', 4, 3, 2]
]

# Derived sets
I = tasks                       # tasks/jobs: 'A','B','C'
J = [1, 2, 3]                   # machines / assembly lines: 1-A, 2-B, 3-C
D = list(range(1, max_construction_days + 1))  # days: 1..3

# Build processing time dictionary p[i,j]
# Table_1_ProcessingTimes: [task_name_with_product, p_on_A, p_on_B, p_on_C]
# machine 1 -> A, 2 -> B, 3 -> C
p = {}
name_to_task = {'A': 'A', 'B': 'B', 'C': 'C'}
for row in Table_1_ProcessingTimes:
    name_with_product, pA, pB, pC = row
    # Extract task key: first character 'A'/'B'/'C'
    task_key = name_with_product.split()[0]  # 'A','B','C'
    i = name_to_task[task_key]
    p[(i, 1)] = pA
    p[(i, 2)] = pB
    p[(i, 3)] = pC

# Big-M for time comparisons (time horizon ≤ 72 hours)
M = 72.0

# =========================
# 2. Model
# =========================

model = gp.Model("China_Europe_Cold_Chain_3Machine_FlowShop")

# =========================
# 3. Decision Variables
# =========================

# Start time of task i on machine j: s_{i,j} in [0,72]
s = model.addVars(
    I, J,
    lb=0.0,
    ub=72.0,
    vtype=GRB.CONTINUOUS,
    name="s"
)

# Makespan: C_max
C_max = model.addVar(
    lb=0.0,
    ub=72.0,
    vtype=GRB.CONTINUOUS,
    name="C_max"
)

# Sequencing variables x_{i,k,j} (for i<k) on each machine j
# x_{i,k,j} = 1 if i precedes k on machine j
x = model.addVars(
    [(i, k, j) for j in J for idx_i, i in enumerate(I)
     for idx_k, k in enumerate(I) if idx_i < idx_k],
    vtype=GRB.BINARY,
    name="x"
)

# Day assignment variables d_{i,j,d}
# d_{i,j,d} = 1 if task i on machine j is processed on day d
dvar = model.addVars(
    I, J, D,
    vtype=GRB.BINARY,
    name="d"
)

model.update()

# =========================
# 4. Objective
# =========================

# Minimize makespan C_max
model.setObjective(C_max, GRB.MINIMIZE)

# =========================
# 5. Constraints
# =========================

# 5.1 Precedence + Transfer Delay
for i in I:
    # s_{i,2} ≥ s_{i,1} + p_{i,1} + transfer_time
    model.addConstr(
        s[(i, 2)] >= s[(i, 1)] + p[(i, 1)] + transfer_time,
        name=f"precedence_transfer_1to2_{i}"
    )
    # s_{i,3} ≥ s_{i,2} + p_{i,2} + transfer_time
    model.addConstr(
        s[(i, 3)] >= s[(i, 2)] + p[(i, 2)] + transfer_time,
        name=f"precedence_transfer_2to3_{i}"
    )

# 5.2 Machine Non-overlap (using indicator constraints, no big-M lin.)
# On each machine j, each pair of tasks i<k cannot overlap
for j in J:
    for idx_i, i in enumerate(I):
        for idx_k, k in enumerate(I):
            if idx_i < idx_k:
                # Indicator: x_{i,k,j} = 1 → i before k on machine j:
                # s_{i,j} + p_{i,j} ≤ s_{k,j}
                model.addGenConstrIndicator(
                    x[(i, k, j)], 1,
                    s[(i, j)] + p[(i, j)] <= s[(k, j)],
                    name=f"nonoverlap_{i}_before_{k}_m{j}"
                )
                # Indicator: x_{i,k,j} = 0 → k before i on machine j:
                # s_{k,j} + p_{k,j} ≤ s_{i,j}
                model.addGenConstrIndicator(
                    x[(i, k, j)], 0,
                    s[(k, j)] + p[(k, j)] <= s[(i, j)],
                    name=f"nonoverlap_{k}_before_{i}_m{j}"
                )

# 5.3 Daily-shift Window and Day Assignment
for i in I:
    for j in J:
        # Exactly one day d for operation (i,j)
        model.addConstr(
            gp.quicksum(dvar[(i, j, d)] for d in D) == 1,
            name=f"day_assign_unique_{i}_{j}"
        )

        for d in D:
            day_start = 24.0 * (d - 1)         # 0, 24, 48
            day_end = day_start + daily_operation_hours  # 14, 38, 62

            # If d_{i,j,d} = 1 → operation (i,j) lies within [day_start, day_end]
            # Indicator for start: d=1 → s_{i,j} ≥ day_start
            model.addGenConstrIndicator(
                dvar[(i, j, d)], 1,
                s[(i, j)] >= day_start,
                name=f"window_start_{i}_{j}_d{d}"
            )

            # Indicator for end: d=1 → s_{i,j} + p_{i,j} ≤ day_end
            model.addGenConstrIndicator(
                dvar[(i, j, d)], 1,
                s[(i, j)] + p[(i, j)] <= day_end,
                name=f"window_end_{i}_{j}_d{d}"
            )

# 5.4 Makespan Definition
for i in I:
    # C_max ≥ s_{i,3} + p_{i,3}
    model.addConstr(
        C_max >= s[(i, 3)] + p[(i, 3)],
        name=f"makespan_def_{i}"
    )

# 5.5 Time Horizon constraint
model.addConstr(
    C_max <= 72.0,
    name="time_horizon"
)

# =========================
# 6. Solve
# =========================

model.setParam('OutputFlag', 0)  # turn off detailed Gurobi log; set to 1 for debugging
model.optimize()

# =========================
# 7. Print Results
# =========================

if model.status == GRB.OPTIMAL:
    print("Optimal solution found.")
    print(f"Optimal makespan (hours from time 0): {C_max.X:.4f}")

    # Detailed schedule
    print("\nStart times s_{i,j} (hours from time 0):")
    for i in I:
        for j in J:
            print(f"Task {i} on machine {j} starts at {s[(i, j)].X:.4f}")

    print("\nDay assignments d_{i,j,d}:")
    for i in I:
        for j in J:
            for d in D:
                if dvar[(i, j, d)].X > 0.5:
                    print(f"Task {i}, machine {j} processed on day {d}")

    # Final answer (the question asks for the minimum total construction period)
    the_question_answer = C_max.X
    print(f"FinalAnswer=【{the_question_answer}】")
else:
    print("No optimal solution found.")
    # If infeasible or other status, still print something for FinalAnswer
    the_question_answer = float('nan')
    print(f"FinalAnswer=【{the_question_answer}】")