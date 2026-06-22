import gurobipy as gp
from gurobipy import GRB

# ====================
# 1. PARAMETERS FROM LIST
# ====================
P = 12  # number_of_phases
J = 6   # number_of_projects
B = 40  # budget_limit_per_period (million yuan)
M = 18  # hr_limit_per_period (people)

# Project data: [cost, profit, manpower]
proj_data = [
    [10, 8, 4],   # Project 1
    [15, 12, 6],  # Project 2
    [8, 7, 3],    # Project 3
    [20, 15, 8],  # Project 4
    [11, 9, 5],   # Project 5
    [9, 8, 4]     # Project 6
]

# Extract lists for convenience
cost = [proj_data[j][0] for j in range(J)]
profit = [proj_data[j][1] for j in range(J)]
manpower = [proj_data[j][2] for j in range(J)]

# Sets
J_RD = [1, 2, 3]           # rd_projects (1‑based indices, but we'll adjust in code)
J_extra = [4, 5, 6]        # unlocking_extra_projects
J_all = list(range(1, J+1))

# Unlocking requirements
unlock_req = {1: 3, 2: 2, 3: 1}  # project -> count needed

# Discount factors
discount_extra = 0.7   # unlocking_discount_factor
discount_synergy = 0.5 # synergy_discount_factor

# R&D limits per period
rd_min = 1  # rd_projects_min_per_period
rd_max = 2  # rd_projects_max_per_period

# Selection limits
max_normal = 4    # max_projects_per_period
max_with_extra = 5  # max_projects_with_incentive

# Big‑M (for z‑variable linearization)
M_big = P + 5

# ====================
# 2. CREATE MODEL
# ====================
model = gp.Model("Shengrui_Investment")

# ====================
# 3. DECISION VARIABLES
# ====================
# x[p][j] = 1 if project j selected normally in period p (1‑based indices)
x = {}
for p in range(1, P+1):
    for j in J_all:
        x[p, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{p}_{j}")

# y[p][j] = 1 if project j selected as extra incentive project in period p (j=4,5,6)
y = {}
for p in range(1, P+1):
    for j in J_extra:
        y[p, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{p}_{j}")

# w[p] = 1 if incentive unlocked before period p
w = {}
for p in range(1, P+1):
    w[p] = model.addVar(vtype=GRB.BINARY, name=f"w_{p}")

# α[p] = 1 if both project 3 and 6 selected normally in period p
alpha = {}
for p in range(1, P+1):
    alpha[p] = model.addVar(vtype=GRB.BINARY, name=f"alpha_{p}")

# s[p] = 1 if cost of project 3 halved in period p (due to synergy in previous period)
s = {}
for p in range(1, P+1):
    s[p] = model.addVar(vtype=GRB.BINARY, name=f"s_{p}")

# z1[p], z2[p], z3[p] = cumulative triggers for projects 1,2,3
z1 = {}
z2 = {}
z3 = {}
for p in range(1, P+1):
    z1[p] = model.addVar(vtype=GRB.BINARY, name=f"z1_{p}")
    z2[p] = model.addVar(vtype=GRB.BINARY, name=f"z2_{p}")
    z3[p] = model.addVar(vtype=GRB.BINARY, name=f"z3_{p}")

model.update()

# ====================
# 4. OBJECTIVE
# ====================
obj = gp.quicksum(
    profit[j-1] * x[p, j] for p in range(1, P+1) for j in J_all
) + gp.quicksum(
    profit[j-1] * y[p, j] for p in range(1, P+1) for j in J_extra
)
model.setObjective(obj, GRB.MAXIMIZE)

# ====================
# 5. CONSTRAINTS
# ====================

# 5.1 Budget per period
for p in range(1, P+1):
    budget_expr = gp.quicksum(cost[j-1] * x[p, j] for j in J_all)
    budget_expr += gp.quicksum(discount_extra * cost[j-1] * y[p, j] for j in J_extra)
    # If s_p = 1, we save half of project 3's cost (cost[2]=8, half=4)
    budget_expr -= 4 * s[p]
    model.addConstr(budget_expr <= B, name=f"budget_{p}")

# 5.2 Manpower per period (only normal projects consume manpower)
for p in range(1, P+1):
    man_expr = gp.quicksum(manpower[j-1] * x[p, j] for j in J_all)
    model.addConstr(man_expr <= M, name=f"manpower_{p}")

# 5.3 Dependency: if project 2 selected, project 1 must be selected
for p in range(1, P+1):
    model.addConstr(x[p, 2] <= x[p, 1], name=f"dep_{p}")

# 5.4 Exclusivity: projects 4 and 5 cannot be implemented simultaneously
for p in range(1, P+1):
    model.addConstr(x[p, 4] + x[p, 5] + y[p, 4] + y[p, 5] <= 1, name=f"excl_{p}")

# 5.5 Period selection limit: normal + extra <= 4 + w_p
for p in range(1, P+1):
    total_sel = gp.quicksum(x[p, j] for j in J_all) + gp.quicksum(y[p, j] for j in J_extra)
    model.addConstr(total_sel <= max_normal + w[p], name=f"sel_limit_{p}")

# 5.6 R&D count per period: at least 1, at most 2 among projects 1,2,3
for p in range(1, P+1):
    rd_sel = gp.quicksum(x[p, j] for j in J_RD)
    model.addConstr(rd_sel >= rd_min, name=f"rd_min_{p}")
    model.addConstr(rd_sel <= rd_max, name=f"rd_max_{p}")

# 5.7 Each project selected at least once yearly (normal or extra)
for j in J_all:
    if j in J_extra:
        expr = gp.quicksum(x[p, j] + y[p, j] for p in range(1, P+1))
    else:
        expr = gp.quicksum(x[p, j] for p in range(1, P+1))
    model.addConstr(expr >= 1, name=f"yearly_min_{j}")

# 5.8 Synergy trigger α_p linearization
for p in range(1, P+1):
    model.addConstr(alpha[p] <= x[p, 3], name=f"alpha1_{p}")
    model.addConstr(alpha[p] <= x[p, 6], name=f"alpha2_{p}")
    model.addConstr(alpha[p] >= x[p, 3] + x[p, 6] - 1, name=f"alpha3_{p}")

# 5.9 Cost halving s_p linearization (for p ≥ 2)
for p in range(2, P+1):
    model.addConstr(s[p] <= alpha[p-1], name=f"cost_halve1_{p}")
    model.addConstr(s[p] <= x[p, 3], name=f"cost_halve2_{p}")
    model.addConstr(s[p] >= alpha[p-1] + x[p, 3] - 1, name=f"cost_halve3_{p}")
# s[1] = 0
model.addConstr(s[1] == 0, name="s1_zero")

# 5.10 Unlock cumulative requirements (these are also implied by yearly constraints, but we keep them for clarity)
model.addConstr(gp.quicksum(x[p, 1] for p in range(1, P+1)) >= 3, name="req1")
model.addConstr(gp.quicksum(x[p, 2] for p in range(1, P+1)) >= 2, name="req2")
model.addConstr(gp.quicksum(x[p, 3] for p in range(1, P+1)) >= 1, name="req3")

# 5.11 Unlock availability w_p definition (for p ≥ 2)
for p in range(2, P+1):
    model.addConstr(w[p] <= z1[p-1], name=f"w1_{p}")
    model.addConstr(w[p] <= z2[p-1], name=f"w2_{p}")
    model.addConstr(w[p] <= z3[p-1], name=f"w3_{p}")
    model.addConstr(w[p] >= z1[p-1] + z2[p-1] + z3[p-1] - 2, name=f"w4_{p}")
# w[1] = 0 (no incentive before first period)
model.addConstr(w[1] == 0, name="w1_zero")

# 5.12 z‑variable linearizations using indicator constraints
for p in range(1, P+1):
    # z1_p: cumulative project 1 >= 3
    cum1 = gp.quicksum(x[t, 1] for t in range(1, p+1))
    model.addGenConstrIndicator(z1[p], 1, cum1 >= 3, name=f"ind_z1_ge_{p}")
    model.addConstr(cum1 <= 2 + M_big * z1[p], name=f"ind_z1_le_{p}")
    
    # z2_p: cumulative project 2 >= 2
    cum2 = gp.quicksum(x[t, 2] for t in range(1, p+1))
    model.addGenConstrIndicator(z2[p], 1, cum2 >= 2, name=f"ind_z2_ge_{p}")
    model.addConstr(cum2 <= 1 + M_big * z2[p], name=f"ind_z2_le_{p}")
    
    # z3_p: cumulative project 3 >= 1
    cum3 = gp.quicksum(x[t, 3] for t in range(1, p+1))
    model.addGenConstrIndicator(z3[p], 1, cum3 >= 1, name=f"ind_z3_ge_{p}")
    model.addConstr(cum3 <= 0 + M_big * z3[p], name=f"ind_z3_le_{p}")

# 5.13 Strategic mandatory: projects 3 and 6 must be selected at least once
# (Already covered by yearly constraint, but we add explicit constraints as per problem)
model.addConstr(gp.quicksum(x[p, 3] + y[p, 3] for p in range(1, P+1)) >= 1, name="mandatory_3")
model.addConstr(gp.quicksum(x[p, 6] + y[p, 6] for p in range(1, P+1)) >= 1, name="mandatory_6")

# ====================
# 6. SOLVE
# ====================
model.optimize()

# ====================
# 7. PRINT RESULTS
# ====================
if model.status == GRB.OPTIMAL:
    print("Optimal solution found.")
    print(f"Objective (total profit) = {model.objVal:.2f} million yuan")
    
    # Print per‑period selections
    for p in range(1, P+1):
        normal_sel = [j for j in J_all if x[p, j].x > 0.5]
        extra_sel = [j for j in J_extra if y[p, j].x > 0.5]
        cost_normal = sum(cost[j-1] for j in normal_sel)
        cost_extra = sum(discount_extra * cost[j-1] for j in extra_sel)
        saving = 4 * s[p].x
        total_cost = cost_normal + cost_extra - saving
        manpower_used = sum(manpower[j-1] for j in normal_sel)
        print(f"Period {p:2d}: normal {normal_sel}, extra {extra_sel}, "
              f"cost={total_cost:.1f}M, manpower={manpower_used}, w={w[p].x:.0f}")
    
    # Cumulative selections
    print("\nCumulative selections (normal):")
    for j in J_all:
        total = sum(x[p, j].x for p in range(1, P+1))
        print(f"  Project {j}: {total:.0f}")
    if len(J_extra) > 0:
        print("Cumulative selections (extra):")
        for j in J_extra:
            total = sum(y[p, j].x for p in range(1, P+1))
            print(f"  Project {j}: {total:.0f}")
    
    # Final answer output as required
    print(f"\nFinalAnswer=【{model.objVal}】")
    
else:
    print("No optimal solution found.")
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible.")
    elif model.status == GRB.UNBOUNDED:
        print("Model is unbounded.")
    else:
        print(f"Optimization ended with status {model.status}")