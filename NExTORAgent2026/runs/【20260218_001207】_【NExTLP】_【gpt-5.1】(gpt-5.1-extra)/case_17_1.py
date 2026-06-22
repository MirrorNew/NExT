import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters (from Parameters List)
# =========================
num_lines = 3
lines = ['A', 'B', 'C']  # A: initial_cutting, B: deep_freezing, C: packaging
num_tasks = 3
tasks = ['A_beef', 'B_cod', 'C_shrimp']
process_order = ['initial_cutting', 'deep_freezing', 'packaging']
transfer_time_between_processes = 0.5
working_start_time = 6
working_end_time = 20
daily_working_hours = 14
max_num_days = 3
max_total_construction_period_hours = 42  # NOTE: original text also mentions 72, but we use given parameter

Table_1_processing_times = [
    {'Task': 'A_beef',
     'Assembly_line_A_initial_cutting_h': 3,
     'Assembly_line_B_deep_freezing_h': 2,
     'Assembly_line_C_packaging_h': 1},
    {'Task': 'B_cod',
     'Assembly_line_A_initial_cutting_h': 2,
     'Assembly_line_B_deep_freezing_h': 4,
     'Assembly_line_C_packaging_h': 1},
    {'Task': 'C_shrimp',
     'Assembly_line_A_initial_cutting_h': 4,
     'Assembly_line_B_deep_freezing_h': 3,
     'Assembly_line_C_packaging_h': 2}
]

# Build convenient mappings
task_ids = ['A', 'B', 'C']  # to match symbols A,B,C
task_name_to_id = {'A_beef': 'A', 'B_cod': 'B', 'C_shrimp': 'C'}
id_to_task_name = {v: k for k, v in task_name_to_id.items()}

# Processing time dict p[task_id][line]
p = {tid: {} for tid in task_ids}
for row in Table_1_processing_times:
    tname = row['Task']
    tid = task_name_to_id[tname]
    p[tid]['A'] = row['Assembly_line_A_initial_cutting_h']
    p[tid]['B'] = row['Assembly_line_B_deep_freezing_h']
    p[tid]['C'] = row['Assembly_line_C_packaging_h']

days = range(1, max_num_days + 1)

# Big-M (used only inside indicator constraints; still must specify)
# Since horizon <= max_total_construction_period_hours,
# pick an M slightly larger than that.
M = 100.0

# =========================
# 2. Create model
# =========================
model = gp.Model("China_Europe_Cold_Chain_3Stage_Scheduling")

# =========================
# 3. Decision Variables
# =========================

# Start and completion times S_{i,k}, C_{i,k}
S = {}
C = {}
for tid in task_ids:
    for line in lines:
        S[tid, line] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                    name=f"S_{tid}_{line}")
        C[tid, line] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                    name=f"C_{tid}_{line}")

# Day indices d_{i,k} ∈ {1,2,3}
d = {}
for tid in task_ids:
    for line in lines:
        d[tid, line] = model.addVar(vtype=GRB.INTEGER, lb=1, ub=max_num_days,
                                    name=f"d_{tid}_{line}")

# Binary ordering variables δ for each pair on each line
# We explicitly create variables with the symbols used in the context.
# Line A
delta_AB_A = model.addVar(vtype=GRB.BINARY, name="delta_AB_A")
delta_BA_A = model.addVar(vtype=GRB.BINARY, name="delta_BA_A")
delta_AC_A = model.addVar(vtype=GRB.BINARY, name="delta_AC_A")
delta_CA_A = model.addVar(vtype=GRB.BINARY, name="delta_CA_A")
delta_BC_A = model.addVar(vtype=GRB.BINARY, name="delta_BC_A")
delta_CB_A = model.addVar(vtype=GRB.BINARY, name="delta_CB_A")

# Line B
delta_AB_B = model.addVar(vtype=GRB.BINARY, name="delta_AB_B")
delta_BA_B = model.addVar(vtype=GRB.BINARY, name="delta_BA_B")
delta_AC_B = model.addVar(vtype=GRB.BINARY, name="delta_AC_B")
delta_CA_B = model.addVar(vtype=GRB.BINARY, name="delta_CA_B")
delta_BC_B = model.addVar(vtype=GRB.BINARY, name="delta_BC_B")
delta_CB_B = model.addVar(vtype=GRB.BINARY, name="delta_CB_B")

# Line C
delta_AB_C = model.addVar(vtype=GRB.BINARY, name="delta_AB_C")
delta_BA_C = model.addVar(vtype=GRB.BINARY, name="delta_BA_C")
delta_AC_C = model.addVar(vtype=GRB.BINARY, name="delta_AC_C")
delta_CA_C = model.addVar(vtype=GRB.BINARY, name="delta_CA_C")
delta_BC_C = model.addVar(vtype=GRB.BINARY, name="delta_BC_C")
delta_CB_C = model.addVar(vtype=GRB.BINARY, name="delta_CB_C")

# Makespan T
T = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="T")

model.update()

# =========================
# 4. Constraints
# =========================

# ---- 4.1 Processing durations ----
# Task A: tid='A'
model.addConstr(C['A', 'A'] == S['A', 'A'] + p['A']['A'], name="Proc_A_lineA")
model.addConstr(C['A', 'B'] == S['A', 'B'] + p['A']['B'], name="Proc_A_lineB")
model.addConstr(C['A', 'C'] == S['A', 'C'] + p['A']['C'], name="Proc_A_lineC")

# Task B: tid='B'
model.addConstr(C['B', 'A'] == S['B', 'A'] + p['B']['A'], name="Proc_B_lineA")
model.addConstr(C['B', 'B'] == S['B', 'B'] + p['B']['B'], name="Proc_B_lineB")
model.addConstr(C['B', 'C'] == S['B', 'C'] + p['B']['C'], name="Proc_B_lineC")

# Task C: tid='C'
model.addConstr(C['C', 'A'] == S['C', 'A'] + p['C']['A'], name="Proc_C_lineA")
model.addConstr(C['C', 'B'] == S['C', 'B'] + p['C']['B'], name="Proc_C_lineB")
model.addConstr(C['C', 'C'] == S['C', 'C'] + p['C']['C'], name="Proc_C_lineC")

# ---- 4.2 Precedence + waiting between lines ----
# Task A
model.addConstr(S['A', 'B'] >= C['A', 'A'] + transfer_time_between_processes,
                name="Prec_wait_A_A_to_B")
model.addConstr(S['A', 'C'] >= C['A', 'B'] + transfer_time_between_processes,
                name="Prec_wait_A_B_to_C")

# Task B
model.addConstr(S['B', 'B'] >= C['B', 'A'] + transfer_time_between_processes,
                name="Prec_wait_B_A_to_B")
model.addConstr(S['B', 'C'] >= C['B', 'B'] + transfer_time_between_processes,
                name="Prec_wait_B_B_to_C")

# Task C
model.addConstr(S['C', 'B'] >= C['C', 'A'] + transfer_time_between_processes,
                name="Prec_wait_C_A_to_B")
model.addConstr(S['C', 'C'] >= C['C', 'B'] + transfer_time_between_processes,
                name="Prec_wait_C_B_to_C")

# ---- 4.3 Machine capacity with indicator constraints ----
# NOTE: We implement the big-M style disjunctions using addGenConstrIndicator,
# as required; we DO NOT add direct big-M linear constraints.

# ===== Line A: pairs (A,B), (A,C), (B,C) =====

# Pair (A,B) on line A, using delta_BA_A
# If delta_BA_A == 1: A after B: S_A_A >= C_B_A
model.addGenConstrIndicator(delta_BA_A, 1,
                            S['A', 'A'] - C['B', 'A'] >= 0,
                            name="Ind_LineA_AB_AafterB")
# If delta_BA_A == 0: B after A: S_B_A >= C_A_A
model.addGenConstrIndicator(delta_BA_A, 0,
                            S['B', 'A'] - C['A', 'A'] >= 0,
                            name="Ind_LineA_AB_BafterA")

# Pair (A,C) on line A, using delta_CA_A
# If delta_CA_A == 1: A after C
model.addGenConstrIndicator(delta_CA_A, 1,
                            S['A', 'A'] - C['C', 'A'] >= 0,
                            name="Ind_LineA_AC_AafterC")
# If delta_CA_A == 0: C after A
model.addGenConstrIndicator(delta_CA_A, 0,
                            S['C', 'A'] - C['A', 'A'] >= 0,
                            name="Ind_LineA_AC_CafterA")

# Pair (B,C) on line A, using delta_CB_A
# If delta_CB_A == 1: B after C
model.addGenConstrIndicator(delta_CB_A, 1,
                            S['B', 'A'] - C['C', 'A'] >= 0,
                            name="Ind_LineA_BC_BafterC")
# If delta_CB_A == 0: C after B
model.addGenConstrIndicator(delta_CB_A, 0,
                            S['C', 'A'] - C['B', 'A'] >= 0,
                            name="Ind_LineA_BC_CafterB")

# ===== Line B: pairs (A,B), (A,C), (B,C) =====

# Pair (A,B) on line B, using delta_BA_B
# If delta_BA_B == 1: A after B
model.addGenConstrIndicator(delta_BA_B, 1,
                            S['A', 'B'] - C['B', 'B'] >= 0,
                            name="Ind_LineB_AB_AafterB")
# If delta_BA_B == 0: B after A
model.addGenConstrIndicator(delta_BA_B, 0,
                            S['B', 'B'] - C['A', 'B'] >= 0,
                            name="Ind_LineB_AB_BafterA")

# Pair (A,C) on line B, using delta_CA_B
# If delta_CA_B == 1: A after C
model.addGenConstrIndicator(delta_CA_B, 1,
                            S['A', 'B'] - C['C', 'B'] >= 0,
                            name="Ind_LineB_AC_AafterC")
# If delta_CA_B == 0: C after A
model.addGenConstrIndicator(delta_CA_B, 0,
                            S['C', 'B'] - C['A', 'B'] >= 0,
                            name="Ind_LineB_AC_CafterA")

# Pair (B,C) on line B, using delta_CB_B
# If delta_CB_B == 1: B after C
model.addGenConstrIndicator(delta_CB_B, 1,
                            S['B', 'B'] - C['C', 'B'] >= 0,
                            name="Ind_LineB_BC_BafterC")
# If delta_CB_B == 0: C after B
model.addGenConstrIndicator(delta_CB_B, 0,
                            S['C', 'B'] - C['B', 'B'] >= 0,
                            name="Ind_LineB_BC_CafterB")

# ===== Line C: pairs (A,B), (A,C), (B,C) =====

# Pair (A,B) on line C, using delta_BA_C
# If delta_BA_C == 1: A after B
model.addGenConstrIndicator(delta_BA_C, 1,
                            S['A', 'C'] - C['B', 'C'] >= 0,
                            name="Ind_LineC_AB_AafterB")
# If delta_BA_C == 0: B after A
model.addGenConstrIndicator(delta_BA_C, 0,
                            S['B', 'C'] - C['A', 'C'] >= 0,
                            name="Ind_LineC_AB_BafterA")

# Pair (A,C) on line C, using delta_CA_C
# If delta_CA_C == 1: A after C
model.addGenConstrIndicator(delta_CA_C, 1,
                            S['A', 'C'] - C['C', 'C'] >= 0,
                            name="Ind_LineC_AC_AafterC")
# If delta_CA_C == 0: C after A
model.addGenConstrIndicator(delta_CA_C, 0,
                            S['C', 'C'] - C['A', 'C'] >= 0,
                            name="Ind_LineC_AC_CafterA")

# Pair (B,C) on line C, using delta_CB_C
# If delta_CB_C == 1: B after C
model.addGenConstrIndicator(delta_CB_C, 1,
                            S['B', 'C'] - C['C', 'C'] >= 0,
                            name="Ind_LineC_BC_BafterC")
# If delta_CB_C == 0: C after B
model.addGenConstrIndicator(delta_CB_C, 0,
                            S['C', 'C'] - C['B', 'C'] >= 0,
                            name="Ind_LineC_BC_CafterB")

# ---- 4.4 Exactly-one ordering per pair on each line ----
# Line A
model.addConstr(delta_AB_A + delta_BA_A == 1, name="ExactlyOne_AB_A")
model.addConstr(delta_AC_A + delta_CA_A == 1, name="ExactlyOne_AC_A")
model.addConstr(delta_BC_A + delta_CB_A == 1, name="ExactlyOne_BC_A")

# Line B
model.addConstr(delta_AB_B + delta_BA_B == 1, name="ExactlyOne_AB_B")
model.addConstr(delta_AC_B + delta_CA_B == 1, name="ExactlyOne_AC_B")
model.addConstr(delta_BC_B + delta_CB_B == 1, name="ExactlyOne_BC_B")

# Line C
model.addConstr(delta_AB_C + delta_BA_C == 1, name="ExactlyOne_AB_C")
model.addConstr(delta_AC_C + delta_CA_C == 1, name="ExactlyOne_AC_C")
model.addConstr(delta_BC_C + delta_CB_C == 1, name="ExactlyOne_BC_C")

# ---- 4.5 Working window constraints ----
for tid in task_ids:
    for line in lines:
        model.addConstr(
            S[tid, line] >= working_start_time + 24 * (d[tid, line] - 1),
            name=f"WorkWindowStart_{tid}_{line}"
        )
        model.addConstr(
            C[tid, line] <= working_end_time + 24 * (d[tid, line] - 1),
            name=f"WorkWindowEnd_{tid}_{line}"
        )

# ---- 4.6 Horizon constraints ----
# Use the provided max_total_construction_period_hours (42)
model.addConstr(T <= max_total_construction_period_hours, name="Horizon_T")
for tid in task_ids:
    for line in lines:
        model.addConstr(
            C[tid, line] <= max_total_construction_period_hours,
            name=f"Horizon_C_{tid}_{line}"
        )

# ---- 4.7 Makespan definition ----
model.addConstr(T >= C['A', 'C'], name="Makespan_A")
model.addConstr(T >= C['B', 'C'], name="Makespan_B")
model.addConstr(T >= C['C', 'C'], name="Makespan_C")

# =========================
# 5. Objective function
# =========================
model.setObjective(T, GRB.MINIMIZE)

# =========================
# 6. Solve model
# =========================
model.optimize()

# =========================
# 7. Output results
# =========================
if model.status == GRB.OPTIMAL:
    print(f"Optimal makespan T = {T.X:.4f} hours")
    print("\nStart and completion times (global hours from 0):")
    for tid in task_ids:
        for line in lines:
            print(f"Task {tid} on line {line}: "
                  f"day={int(round(d[tid, line].X))}, "
                  f"S={S[tid, line].X:.2f}, C={C[tid, line].X:.2f}")

    print("\nOrdering decisions on Line A:")
    print(f"delta_AB_A={int(round(delta_AB_A.X))}, delta_BA_A={int(round(delta_BA_A.X))}")
    print(f"delta_AC_A={int(round(delta_AC_A.X))}, delta_CA_A={int(round(delta_CA_A.X))}")
    print(f"delta_BC_A={int(round(delta_BC_A.X))}, delta_CB_A={int(round(delta_CB_A.X))}")

    print("\nOrdering decisions on Line B:")
    print(f"delta_AB_B={int(round(delta_AB_B.X))}, delta_BA_B={int(round(delta_BA_B.X))}")
    print(f"delta_AC_B={int(round(delta_AC_B.X))}, delta_CA_B={int(round(delta_CA_B.X))}")
    print(f"delta_BC_B={int(round(delta_BC_B.X))}, delta_CB_B={int(round(delta_CB_B.X))}")

    print("\nOrdering decisions on Line C:")
    print(f"delta_AB_C={int(round(delta_AB_C.X))}, delta_BA_C={int(round(delta_BA_C.X))}")
    print(f"delta_AC_C={int(round(delta_AC_C.X))}, delta_CA_C={int(round(delta_CA_C.X))}")
    print(f"delta_BC_C={int(round(delta_BC_C.X))}, delta_CB_C={int(round(delta_CB_C.X))}")

# =========================
# 8. Final answer (as required)
# =========================
# The question asks for the minimum total construction period for all completions,
# which is the optimal makespan T.
final_answer_value = T.X if model.status == GRB.OPTIMAL else float('nan')
print(f"FinalAnswer=【{final_answer_value}】")