import gurobipy as gp
from gurobipy import GRB

# ===============================
# 1. Parameters (from Parameters List)
# ===============================
number_of_phases = 12
number_of_projects = 6
budget_limit_per_period = 40
hr_limit_per_period = 18
resources_reset_each_phase = True
synergy_pair = [3, 6]
synergy_discount_factor = 0.5
synergy_duration_phases = 1
unlocking_requirements = {'1': 3, '2': 2, '3': 1}
unlocking_extra_projects = [4, 5, 6]
unlocking_discount_factor = 0.7
unlocking_ignore_hr_constraint = True
exclusivity_pairs = [[4, 5]]
dependency = {'2': ['1']}
minimal_selection_each_project = {'1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1}
max_projects_per_period = 4
max_projects_with_incentive = 5
rd_projects = [1, 2, 3]
rd_projects_min_per_period = 1
rd_projects_max_per_period = 2
strategic_mandatory_projects = [3, 6]
Table_1_ProjectInfo = [
    ['Project 1', 'R&D', 10, 8, 4],
    ['Project 2', 'R&D', 15, 12, 6],
    ['Project 3', 'R&D', 8, 7, 3],
    ['Project 4', 'Implementation', 20, 15, 8],
    ['Project 5', 'Implementation', 11, 9, 5],
    ['Project 6', 'Implementation', 9, 8, 4]
]

# Extract cost, profit, manpower from table
projects = range(1, number_of_projects + 1)
phases = range(1, number_of_phases + 1)

cost = {}
profit = {}
manpower = {}
project_type = {}

for j_idx, row in enumerate(Table_1_ProjectInfo, start=1):
    name, ptype, c, pi, m = row
    cost[j_idx] = c
    profit[j_idx] = pi
    manpower[j_idx] = m
    project_type[j_idx] = ptype

# Constants
B = budget_limit_per_period
Hlim = hr_limit_per_period
M_big = number_of_phases  # big-M for cumulative constraints (12 is enough)

# ===============================
# 2. Create model
# ===============================
model = gp.Model("Shengrui_12_Phase_Project_Investment")

# ===============================
# 3. Decision variables
# ===============================

# x[p,j] = 1 if project j selected in phase p under normal rules
x = model.addVars(phases, projects, vtype=GRB.BINARY, name="x")

# y[p,j] = 1 if project j in {4,5,6} selected in phase p as extra incentive project
y = model.addVars(phases, unlocking_extra_projects, vtype=GRB.BINARY, name="y")

# w[p] = 1 if incentives unlocked before (i.e., available in) phase p
w = model.addVars(phases, vtype=GRB.BINARY, name="w")

# z1[p], z2[p], z3[p] - cumulative threshold flags for projects 1, 2, 3
z1 = model.addVars(phases, vtype=GRB.BINARY, name="z1")
z2 = model.addVars(phases, vtype=GRB.BINARY, name="z2")
z3 = model.addVars(phases, vtype=GRB.BINARY, name="z3")

# alpha[p] = 1 if in phase p both project 3 and 6 selected (normal x)
alpha = model.addVars(phases, vtype=GRB.BINARY, name="alpha")

# s[p] = 1 if cost of project 3 halved in phase p (defined only for p>=2)
s = model.addVars(range(2, number_of_phases + 1), vtype=GRB.BINARY, name="s")

# ===============================
# 4. Objective function
# Maximize total profit over the year
# ===============================
obj = gp.quicksum(
    profit[j] * x[p, j] for p in phases for j in projects
) + gp.quicksum(
    profit[j] * y[p, j] for p in phases for j in unlocking_extra_projects
)

model.setObjective(obj, GRB.MAXIMIZE)

# ===============================
# 5. Constraints
# ===============================

# 5.1 Budget per period
for p in phases:
    # base cost: sum_j c_j * x[p,j]
    expr = gp.quicksum(cost[j] * x[p, j] for j in projects)
    # incentive projects with discount 0.7 on cost
    expr += gp.quicksum(unlocking_discount_factor * cost[j] * y[p, j]
                        for j in unlocking_extra_projects)
    # cost-halving adjustment for project 3: subtract 4*s_p when p>=2
    if p >= 2:
        expr -= 4 * s[p]
    # for p=1, s_1 is treated as 0 (no variable)
    model.addConstr(expr <= B, name=f"Budget_{p}")

# 5.2 Manpower per period (only normal selections count)
for p in phases:
    model.addConstr(
        gp.quicksum(manpower[j] * x[p, j] for j in projects) <= Hlim,
        name=f"Manpower_{p}"
    )

# 5.3 Dependency: project 2 requires project 1 in same period
for p in phases:
    model.addConstr(x[p, 2] <= x[p, 1], name=f"Dep_2_on_1_p{p}")

# 5.4 Exclusivity between project 4 and 5 (x and y)
for p in phases:
    model.addConstr(
        x[p, 4] + x[p, 5] + y[p, 4] + y[p, 5] <= 1,
        name=f"Excl_4_5_p{p}"
    )

# 5.5 Period selection limit: total projects <= 4 + w_p
for p in phases:
    model.addConstr(
        gp.quicksum(x[p, j] for j in projects) +
        gp.quicksum(y[p, j] for j in unlocking_extra_projects)
        <= max_projects_per_period + w[p],
        name=f"PeriodSelLimit_p{p}"
    )
    # At most one extra incentive project in a period
    model.addConstr(
        gp.quicksum(y[p, j] for j in unlocking_extra_projects) <= w[p],
        name=f"ExtraIncentiveLimit_p{p}"
    )

# 5.6 R&D count per period: 1 <= sum_{j in R&D} x[p,j] <= 2
for p in phases:
    rd_sum = gp.quicksum(x[p, j] for j in rd_projects)
    model.addConstr(rd_sum >= rd_projects_min_per_period, name=f"RDmin_p{p}")
    model.addConstr(rd_sum <= rd_projects_max_per_period, name=f"RDmax_p{p}")

# 5.7 Annual selection: each project >=1 over 12 periods
for j in projects:
    if j in unlocking_extra_projects:
        # both normal and incentive selections count
        model.addConstr(
            gp.quicksum(x[p, j] for p in phases) +
            gp.quicksum(y[p, j] for p in phases)
            >= minimal_selection_each_project[str(j)],
            name=f"AnnualMin_j{j}"
        )
    else:
        # only normal x selections for R&D projects 1..3
        model.addConstr(
            gp.quicksum(x[p, j] for p in phases)
            >= minimal_selection_each_project[str(j)],
            name=f"AnnualMin_j{j}"
        )

# (Also satisfies strategic_mandatory_projects {3,6} automatically.)

# 5.8 Synergy-trigger: alpha_p = 1 iff x[p,3]=x[p,6]=1
for p in phases:
    model.addConstr(alpha[p] <= x[p, 3], name=f"Alpha_le_x3_p{p}")
    model.addConstr(alpha[p] <= x[p, 6], name=f"Alpha_le_x6_p{p}")
    model.addConstr(alpha[p] >= x[p, 3] + x[p, 6] - 1, name=f"Alpha_ge_sum_p{p}")

# 5.9 Cost-halve for project 3 (p >= 2):
# s_p = 1 iff alpha_{p-1} = 1 AND x[p,3] = 1
for p in range(2, number_of_phases + 1):
    model.addConstr(s[p] <= alpha[p - 1], name=f"s_le_alpha_prev_p{p}")
    model.addConstr(s[p] <= x[p, 3], name=f"s_le_x3_p{p}")
    model.addConstr(
        s[p] >= alpha[p - 1] + x[p, 3] - 1,
        name=f"s_ge_alpha_prev_plus_x3_minus1_p{p}"
    )

# 5.10 Unlock-count requirements global (overall feasibility)
model.addConstr(
    gp.quicksum(x[p, 1] for p in phases) >= unlocking_requirements['1'],
    name="UnlockReq_Proj1"
)
model.addConstr(
    gp.quicksum(x[p, 2] for p in phases) >= unlocking_requirements['2'],
    name="UnlockReq_Proj2"
)
model.addConstr(
    gp.quicksum(x[p, 3] for p in phases) >= unlocking_requirements['3'],
    name="UnlockReq_Proj3"
)

# 5.11 z-variable linearizations (indicator constraints instead of big-M)
# For each p, we enforce:
#   z1_p = 1  <=>  cumulative x[1..p,1] >= 3
#   z2_p = 1  <=>  cumulative x[1..p,2] >= 2
#   z3_p = 1  <=>  cumulative x[1..p,3] >= 1
for p in phases:
    cum1 = gp.quicksum(x[t, 1] for t in range(1, p + 1))
    cum2 = gp.quicksum(x[t, 2] for t in range(1, p + 1))
    cum3 = gp.quicksum(x[t, 3] for t in range(1, p + 1))

    # Project 1 threshold: >=3
    # If z1_p = 1 then cum1 >= 3
    model.addGenConstrIndicator(z1[p], 1, cum1 >= unlocking_requirements['1'],
                                name=f"z1_{p}_ind1")
    # If z1_p = 0 then cum1 <= 2
    model.addGenConstrIndicator(z1[p], 0, cum1 <= unlocking_requirements['1'] - 1,
                                name=f"z1_{p}_ind0")

    # Project 2 threshold: >=2
    model.addGenConstrIndicator(z2[p], 1, cum2 >= unlocking_requirements['2'],
                                name=f"z2_{p}_ind1")
    model.addGenConstrIndicator(z2[p], 0, cum2 <= unlocking_requirements['2'] - 1,
                                name=f"z2_{p}_ind0")

    # Project 3 threshold: >=1
    model.addGenConstrIndicator(z3[p], 1, cum3 >= unlocking_requirements['3'],
                                name=f"z3_{p}_ind1")
    model.addGenConstrIndicator(z3[p], 0, cum3 <= unlocking_requirements['3'] - 1,
                                name=f"z3_{p}_ind0")

# 5.12 Unlock-availability definition w_p (p>=2)
# w_p = 1 iff z1_{p-1}=1 and z2_{p-1}=1 and z3_{p-1}=1
for p in phases:
    if p == 1:
        # No incentive available before first period
        model.addConstr(w[p] == 0, name="w_1_zero")
    else:
        model.addConstr(w[p] <= z1[p - 1], name=f"w_le_z1_prev_p{p}")
        model.addConstr(w[p] <= z2[p - 1], name=f"w_le_z2_prev_p{p}")
        model.addConstr(w[p] <= z3[p - 1], name=f"w_le_z3_prev_p{p}")
        model.addConstr(
            w[p] >= z1[p - 1] + z2[p - 1] + z3[p - 1] - 2,
            name=f"w_ge_sum_prev_minus2_p{p}"
        )

# ===============================
# 6. Optimize
# ===============================
model.optimize()

# ===============================
# 7. Print results
# ===============================
if model.Status == GRB.OPTIMAL:
    total_profit = model.ObjVal
    print("Optimal total profit:", total_profit)

    # Optional: print schedule
    for p in phases:
        print(f"\nPhase {p}:")
        chosen_normal = [j for j in projects if x[p, j].X > 0.5]
        chosen_incentive = [j for j in unlocking_extra_projects if y[p, j].X > 0.5]
        print("  Normal projects:", chosen_normal)
        print("  Incentive projects:", chosen_incentive)
        print("  w (unlock available):", int(round(w[p].X)))
        if p >= 2:
            print("  s (cost halved for proj3):", int(round(s[p].X)))
        print("  alpha (3 & 6 synergy):", int(round(alpha[p].X)))

    # Final answer: only the total profit is required
    print(f"FinalAnswer=【{total_profit}】")
else:
    # Infeasible or other status
    print("No optimal solution found. Status code:", model.Status)
    # For consistency, output something (e.g., 0) as FinalAnswer
    print("FinalAnswer=【0】")