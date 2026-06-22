import gurobipy as gp

# Solve the Shengrui 12-phase investment MILP
# Question: maximize total annual profit (million yuan) subject to all given rules

# ===============================
# 1. Define parameters (from Parameters List)
# ===============================
number_of_phases = 12
phases_per_year = 12
number_of_projects = 6

period_budget_max = 40.0
max_investment_per_phase = 40.0  # not used separately, same value

period_manpower_max = 18
max_manpower_per_phase = 18      # not used separately, same value

p3_p6_cost_discount_factor_next_phase = 0.5  # P3 cost halved next phase when synergy P3+P6
p3_cost_restoration_factor_after = 1.0       # conceptually restores to base 8

incentive_trigger_p1 = 3
incentive_trigger_p2 = 2
incentive_trigger_p3 = 1

incentive_extra_projects_per_period = 1
incentive_extra_project_set = [4, 5, 6]
incentive_cost_discount_factor = 0.7
incentive_extra_project_hr_constrained = 0   # 0 => incentive project not counted in HR

p4_p5_mutual_exclusion_same_period = 1
dependency_p2_requires_p1 = 1

yearly_min_selection_p3 = 1
yearly_min_selection_p6 = 1
yearly_min_selection_each_project = 1

max_projects_per_period_base = 4
max_projects_per_period_with_incentive = 5   # logically 4 + 1, implicit via u_t and y

min_RnD_projects_per_period = 1
max_RnD_projects_per_period = 2

RnD_projects = [1, 2, 3]

# Table_1 project data
Table_1_ProjectData_project_no = [1, 2, 3, 4, 5, 6]
Table_1_ProjectData_project_name = [
    'Project 1', 'Project 2', 'Project 3',
    'Project 4', 'Project 5', 'Project 6'
]
Table_1_ProjectData_category = [
    'R&D', 'R&D', 'R&D',
    'Implementation', 'Implementation', 'Implementation'
]
Table_1_ProjectData_cost_million = [10.0, 15.0, 8.0, 20.0, 11.0, 9.0]
Table_1_ProjectData_profit_million = [8.0, 12.0, 7.0, 15.0, 9.0, 8.0]
Table_1_ProjectData_manpower = [4, 6, 3, 8, 5, 4]

# Convenience sets (1-based indexing for projects/phases)
phases = range(1, number_of_phases + 1)
projects = range(1, number_of_projects + 1)

cost = {i: Table_1_ProjectData_cost_million[i - 1] for i in projects}
profit = {i: Table_1_ProjectData_profit_million[i - 1] for i in projects}
hr = {i: Table_1_ProjectData_manpower[i - 1] for i in projects}


# ===============================
# 2. Create model
# ===============================
model = gp.Model("Shengrui_12Phase_Investment")


# ===============================
# 3. Decision variables
# ===============================

# x[i,t] = 1 if project i is selected as normal project in phase t
x = model.addVars(projects, phases, vtype=gp.GRB.BINARY, name="x")

# y[k,t] = 1 if project k in incentive set {4,5,6} is selected as extra incentive project in phase t
incentive_projects = incentive_extra_project_set  # [4,5,6]
y = model.addVars(incentive_projects, phases, vtype=gp.GRB.BINARY, name="y")

# u[t] = 1 if incentive is available in phase t
u = model.addVars(phases, vtype=gp.GRB.BINARY, name="u")

# z[t] = 1 if both project 3 and project 6 are selected normally in phase t (synergy)
z = model.addVars(range(1, number_of_phases), vtype=gp.GRB.BINARY, name="z")  # t = 1..11

# c3[t] = effective unit cost of Project 3 in phase t, 4 <= c3_t <= 8
c3 = model.addVars(phases, vtype=gp.GRB.CONTINUOUS, lb=4.0, ub=8.0, name="c3")

# Yearly totals (normal selections of P1,P2,P3) – used conceptually, not strictly needed but kept
N1 = model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=phases_per_year, name="N1")
N2 = model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=phases_per_year, name="N2")
N3 = model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=phases_per_year, name="N3")

# Prefix cumulative counts C1_t, C2_t, C3_t
C1 = model.addVars(phases, vtype=gp.GRB.INTEGER, lb=0, ub=phases_per_year, name="C1")
C2 = model.addVars(phases, vtype=gp.GRB.INTEGER, lb=0, ub=phases_per_year, name="C2")
C3 = model.addVars(phases, vtype=gp.GRB.INTEGER, lb=0, ub=phases_per_year, name="C3")

# v[t] = 1 if unlocking conditions are satisfied by end of phase t
v = model.addVars(phases, vtype=gp.GRB.BINARY, name="v")


# ===============================
# 4. Objective: maximize total annual profit
# ===============================
model.setObjective(
    gp.quicksum(
        profit[1] * x[1, t] +
        profit[2] * x[2, t] +
        profit[3] * x[3, t] +
        profit[4] * (x[4, t] + (y[4, t] if 4 in incentive_projects else 0)) +
        profit[5] * (x[5, t] + (y[5, t] if 5 in incentive_projects else 0)) +
        profit[6] * (x[6, t] + (y[6, t] if 6 in incentive_projects else 0))
        for t in phases
    ),
    gp.GRB.MAXIMIZE
)


# ===============================
# 5. Constraints
# ===============================

# 5.1 Per-phase budget limit
for t in phases:
    expr_budget = (
        cost[1] * x[1, t] +
        cost[2] * x[2, t] +
        c3[t] * x[3, t] +
        cost[4] * x[4, t] +
        cost[5] * x[5, t] +
        cost[6] * x[6, t]
    )
    # Incentive projects: discounted cost
    if 4 in incentive_projects:
        expr_budget += incentive_cost_discount_factor * cost[4] * y[4, t]
    if 5 in incentive_projects:
        expr_budget += incentive_cost_discount_factor * cost[5] * y[5, t]
    if 6 in incentive_projects:
        expr_budget += incentive_cost_discount_factor * cost[6] * y[6, t]

    model.addConstr(expr_budget <= period_budget_max, name=f"Budget_{t}")

# 5.2 Per-phase manpower limit (incentive projects are not HR-constrained)
for t in phases:
    expr_hr = (
        hr[1] * x[1, t] +
        hr[2] * x[2, t] +
        hr[3] * x[3, t] +
        hr[4] * x[4, t] +
        hr[5] * x[5, t] +
        hr[6] * x[6, t]
    )
    model.addConstr(expr_hr <= period_manpower_max, name=f"HR_{t}")

# 5.3 Dependency: Project 2 requires Project 1 in the same phase
if dependency_p2_requires_p1 == 1:
    for t in phases:
        model.addConstr(x[2, t] <= x[1, t], name=f"Dep_P2_P1_{t}")

# 5.4 Mutual exclusion: Projects 4 and 5 (normal + incentive)
if p4_p5_mutual_exclusion_same_period == 1:
    for t in phases:
        expr_p4p5 = x[4, t] + x[5, t]
        if 4 in incentive_projects:
            expr_p4p5 += y[4, t]
        if 5 in incentive_projects:
            expr_p4p5 += y[5, t]
        model.addConstr(expr_p4p5 <= 1, name=f"Excl_P4_P5_{t}")

# 5.5 Annual minimum selection requirements
# Project 1: normal only
model.addConstr(
    gp.quicksum(x[1, t] for t in phases) >= yearly_min_selection_each_project,
    name="Yearly_min_P1"
)
# Project 2: normal only
model.addConstr(
    gp.quicksum(x[2, t] for t in phases) >= yearly_min_selection_each_project,
    name="Yearly_min_P2"
)
# Project 3: normal only
model.addConstr(
    gp.quicksum(x[3, t] for t in phases) >= yearly_min_selection_p3,
    name="Yearly_min_P3"
)
# Project 4: normal or incentive
model.addConstr(
    gp.quicksum(x[4, t] for t in phases) +
    (gp.quicksum(y[4, t] for t in phases) if 4 in incentive_projects else 0)
    >= yearly_min_selection_each_project,
    name="Yearly_min_P4"
)
# Project 5: normal or incentive
model.addConstr(
    gp.quicksum(x[5, t] for t in phases) +
    (gp.quicksum(y[5, t] for t in phases) if 5 in incentive_projects else 0)
    >= yearly_min_selection_each_project,
    name="Yearly_min_P5"
)
# Project 6: normal or incentive
model.addConstr(
    gp.quicksum(x[6, t] for t in phases) +
    (gp.quicksum(y[6, t] for t in phases) if 6 in incentive_projects else 0)
    >= yearly_min_selection_p6,
    name="Yearly_min_P6"
)

# 5.6 Maximum number of normal projects per period
for t in phases:
    model.addConstr(
        gp.quicksum(x[i, t] for i in projects) <= max_projects_per_period_base,
        name=f"Max_normal_proj_{t}"
    )

# 5.7 At most 1 incentive project per period when unlocked
for t in phases:
    model.addConstr(
        gp.quicksum(y[k, t] for k in incentive_projects)
        <= incentive_extra_projects_per_period * u[t],
        name=f"Max_incentive_proj_{t}"
    )

# 5.8 At least 1 R&D project per period
for t in phases:
    model.addConstr(
        gp.quicksum(x[i, t] for i in RnD_projects) >= min_RnD_projects_per_period,
        name=f"Min_RnD_{t}"
    )

# 5.9 At most 2 R&D projects per period
for t in phases:
    model.addConstr(
        gp.quicksum(x[i, t] for i in RnD_projects) <= max_RnD_projects_per_period,
        name=f"Max_RnD_{t}"
    )

# 5.10 Synergy indicator z[t] for projects 3 and 6 (normal only)
for t in range(1, number_of_phases):  # t = 1..11
    model.addConstr(z[t] <= x[3, t], name=f"Synergy_leqP3_{t}")
    model.addConstr(z[t] <= x[6, t], name=f"Synergy_leqP6_{t}")
    model.addConstr(z[t] >= x[3, t] + x[6, t] - 1, name=f"Synergy_ge_sum_{t}")

# 5.11 Project 3 cost dynamics with synergy
# Initial cost: c3[1] = base cost (8)
model.addConstr(c3[1] == cost[3], name="P3_cost_init")

# For t = 1..11:
#   if z[t] = 0 => c3[t+1] = 8
#   if z[t] = 1 => c3[t+1] = 4
# Implement with two-sided linear constraints coherent with the validated math:
#   c3_{t+1} <= 8 - 4*z[t]
#   c3_{t+1} >= 4*z[t]
for t in range(1, number_of_phases):
    # Upper bound: when z[t]=0, c3[t+1] <= 8; when z[t]=1, c3[t+1] <= 4
    model.addConstr(
        c3[t + 1] <= cost[3] - (cost[3] - cost[3] * p3_p6_cost_discount_factor_next_phase) * z[t],
        name=f"P3_cost_upper_{t}"
    )
    # Lower bound: when z[t]=0, c3[t+1] >= 0 but domain [4,8] forces it to 8; when z[t]=1, c3[t+1] >= 4
    model.addConstr(
        c3[t + 1] >= cost[3] * p3_p6_cost_discount_factor_next_phase * z[t],
        name=f"P3_cost_lower_{t}"
    )
    # Domain [4,8] already imposed via variable bounds

# 5.12 Definitions of N1, N2, N3 (yearly counts)
model.addConstr(N1 == gp.quicksum(x[1, t] for t in phases), name="Def_N1")
model.addConstr(N2 == gp.quicksum(x[2, t] for t in phases), name="Def_N2")
model.addConstr(N3 == gp.quicksum(x[3, t] for t in phases), name="Def_N3")

# 5.13 Prefix counts C1_t, C2_t, C3_t
for t in phases:
    model.addConstr(
        C1[t] == gp.quicksum(x[1, tau] for tau in range(1, t + 1)),
        name=f"Def_C1_{t}"
    )
    model.addConstr(
        C2[t] == gp.quicksum(x[2, tau] for tau in range(1, t + 1)),
        name=f"Def_C2_{t}"
    )
    model.addConstr(
        C3[t] == gp.quicksum(x[3, tau] for tau in range(1, t + 1)),
        name=f"Def_C3_{t}"
    )

# 5.14 Unlocking condition satisfaction indicator v[t]
for t in phases:
    model.addConstr(C1[t] >= incentive_trigger_p1 * v[t], name=f"Unlock_P1_{t}")
    model.addConstr(C2[t] >= incentive_trigger_p2 * v[t], name=f"Unlock_P2_{t}")
    model.addConstr(C3[t] >= incentive_trigger_p3 * v[t], name=f"Unlock_P3_{t}")

# 5.15 Incentive availability after unlocking: u[t] <= v[t-1], with v[0] = 0
v0 = 0
for t in phases:
    if t == 1:
        model.addConstr(u[t] <= v0, name="Incentive_phase1")
    else:
        model.addConstr(u[t] <= v[t - 1], name=f"Incentive_after_unlock_{t}")


# ===============================
# 6. Optimize model
# ===============================
model.optimize()


# ===============================
# 7. Print results (including FinalAnswer)
# ===============================
if model.status == gp.GRB.OPTIMAL:
    total_profit = model.objVal
    print(f"Optimal total annual profit (million yuan): {total_profit:.2f}")

    # Optional: print schedule
    for t in phases:
        chosen_normal = [i for i in projects if x[i, t].X > 0.5]
        chosen_incentive = [k for k in incentive_projects if y[k, t].X > 0.5]
        if chosen_normal or chosen_incentive:
            print(f"Phase {t}:")
            print("  Normal projects:", chosen_normal if chosen_normal else "None")
            print("  Incentive project(s):", chosen_incentive if chosen_incentive else "None")

    # Required output format for the question's answer (maximized profit)
    print(f"FinalAnswer=【{total_profit:.2f}】")
else:
    print(f"Model did not solve to optimality. Status code: {model.status}")
    # In case of no optimal solution, still print something for FinalAnswer
    print("FinalAnswer=【NaN】")