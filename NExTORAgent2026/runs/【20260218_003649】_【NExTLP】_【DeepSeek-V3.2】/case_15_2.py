import gurobipy as gp
from gurobipy import GRB

# ====================
# 1. PARAMETERS
# ====================
P = 12  # number_of_phases
J = 6   # number_of_projects
B = 40  # budget_limit_per_period
M = 18  # hr_limit_per_period

# Project data from Table_1_ProjectInfo
cost = [10, 15, 8, 20, 11, 9]       # million yuan
profit = [8, 12, 7, 15, 9, 8]       # million yuan
manpower = [4, 6, 3, 8, 5, 4]       # people

# Sets
J_RD = [1, 2, 3]                    # rd_projects
J_extra = [4, 5, 6]                 # unlocking_extra_projects
J_all = list(range(1, J+1))

# Discount factors
discount_extra = 0.7                # unlocking_discount_factor
synergy_saving = 4                  # cost[2] * synergy_discount_factor = 8 * 0.5 = 4

# Limits
rd_min = 1                          # rd_projects_min_per_period
rd_max = 2                          # rd_projects_max_per_period
max_normal = 4                      # max_projects_per_period

# Unlocking requirements
unlock_req = {1: 3, 2: 2, 3: 1}    # unlocking_requirements

# Big-M constant
M_big = P + 5

# ====================
# 2. CREATE MODEL
# ====================
model = gp.Model("Shengrui_Investment")

# ====================
# 3. DECISION VARIABLES
# ====================
# Binary variables
x = {}
for p in range(1, P+1):
    for j in J_all:
        x[p, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{p}_{j}")

y = {}
for p in range(1, P+1):
    for j in J_extra:
        y[p, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{p}_{j}")

w = {}
for p in range(1, P+1):
    w[p] = model.addVar(vtype=GRB.BINARY, name=f"w_{p}")

alpha = {}
for p in range(1, P+1):
    alpha[p] = model.addVar(vtype=GRB.BINARY, name=f"alpha_{p}")

s = {}
for p in range(1, P+1):
    s[p] = model.addVar(vtype=GRB.BINARY, name=f"s_{p}")

z1 = {}
z2 = {}
z3 = {}
for p in range(1, P+1):
    z1[p] = model.addVar(vtype=GRB.BINARY, name=f"z1_{p}")
    z2[p] = model.addVar(vtype=GRB.BINARY, name=f"z2_{p}")
    z3[p] = model.addVar(vtype=GRB.BINARY, name=f"z3_{p}")

model.update()

# ====================
# 4. OBJECTIVE FUNCTION
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
    budget_expr -= synergy_saving * s[p]
    model.addConstr(budget_expr <= B, name=f"budget_{p}")

# 5.2 Manpower per period
for p in range(1, P+1):
    man_expr = gp.quicksum(manpower[j-1] * x[p, j] for j in J_all)
    model.addConstr(man_expr <= M, name=f"manpower_{p}")

# 5.3 Dependency: x_p2 ≤ x_p1
for p in range(1, P+1):
    model.addConstr(x[p, 2] <= x[p, 1], name=f"dep_{p}")

# 5.4 Exclusivity: x_p4 + x_p5 + y_p4 + y_p5 ≤ 1
for p in range(1, P+1):
    model.addConstr(x[p, 4] + x[p, 5] + y[p, 4] + y[p, 5] <= 1, name=f"excl_{p}")

# 5.5 Period selection limit: normal + extra ≤ 4 + w_p
for p in range(1, P+1):
    total_sel = gp.quicksum(x[p, j] for j in J_all) + gp.quicksum(y[p, j] for j in J_extra)
    model.addConstr(total_sel <= max_normal + w[p], name=f"sel_limit_{p}")

# 5.6 R&D count per period: 1 ≤ ∑_{j=1..3} x_pj ≤ 2
for p in range(1, P+1):
    rd_sel = gp.quicksum(x[p, j] for j in J_RD)
    model.addConstr(rd_sel >= rd_min, name=f"rd_min_{p}")
    model.addConstr(rd_sel <= rd_max, name=f"rd_max_{p}")

# 5.7 Each project selected at least once yearly
for j in J_all:
    if j in J_extra:
        expr = gp.quicksum(x[p, j] + y[p, j] for p in range(1, P+1))
    else:
        expr = gp.quicksum(x[p, j] for p in range(1, P+1))
    model.addConstr(expr >= 1, name=f"yearly_min_{j}")

# 5.8 Synergy trigger alpha_p linearization
for p in range(1, P+1):
    model.addConstr(alpha[p] <= x[p, 3], name=f"alpha1_{p}")
    model.addConstr(alpha[p] <= x[p, 6], name=f"alpha2_{p}")
    model.addConstr(alpha[p] >= x[p, 3] + x[p, 6] - 1, name=f"alpha3_{p}")

# 5.9 Cost halving s_p linearization (for p ≥ 2)
for p in range(2, P+1):
    model.addConstr(s[p] <= alpha[p-1], name=f"cost_halve1_{p}")
    model.addConstr(s[p] <= x[p, 3], name=f"cost_halve2_{p}")
    model.addConstr(s[p] >= alpha[p-1] + x[p, 3] - 1, name=f"cost_halve3_{p}")
model.addConstr(s[1] == 0, name="s1_zero")

# 5.10 Unlock cumulative requirements
model.addConstr(gp.quicksum(x[p, 1] for p in range(1, P+1)) >= unlock_req[1], name="req1")
model.addConstr(gp.quicksum(x[p, 2] for p in range(1, P+1)) >= unlock_req[2], name="req2")
model.addConstr(gp.quicksum(x[p, 3] for p in range(1, P+1)) >= unlock_req[3], name="req3")

# 5.11 Unlock availability w_p definition (for p ≥ 2)
for p in range(2, P+1):
    model.addConstr(w[p] <= z1[p-1], name=f"w1_{p}")
    model.addConstr(w[p] <= z2[p-1], name=f"w2_{p}")
    model.addConstr(w[p] <= z3[p-1], name=f"w3_{p}")
    model.addConstr(w[p] >= z1[p-1] + z2[p-1] + z3[p-1] - 2, name=f"w4_{p}")
model.addConstr(w[1] == 0, name="w1_zero")

# 5.12 z-variable linearizations using indicator constraints
for p in range(1, P+1):
    # z1_p: cumulative project 1 >= 3
    cum1 = gp.quicksum(x[t, 1] for t in range(1, p+1))
    model.addGenConstrIndicator(z1[p], 1, cum1 >= unlock_req[1], name=f"ind_z1_ge_{p}")
    model.addConstr(cum1 <= (unlock_req[1] - 1) + M_big * z1[p], name=f"ind_z1_le_{p}")
    
    # z2_p: cumulative project 2 >= 2
    cum2 = gp.quicksum(x[t, 2] for t in range(1, p+1))
    model.addGenConstrIndicator(z2[p], 1, cum2 >= unlock_req[2], name=f"ind_z2_ge_{p}")
    model.addConstr(cum2 <= (unlock_req[2] - 1) + M_big * z2[p], name=f"ind_z2_le_{p}")
    
    # z3_p: cumulative project 3 >= 1
    cum3 = gp.quicksum(x[t, 3] for t in range(1, p+1))
    model.addGenConstrIndicator(z3[p], 1, cum3 >= unlock_req[3], name=f"ind_z3_ge_{p}")
    model.addConstr(cum3 <= (unlock_req[3] - 1) + M_big * z3[p], name=f"ind_z3_le_{p}")

# 5.13 Strategic mandatory projects
model.addConstr(gp.quicksum(x[p, 3] + y[p, 3] for p in range(1, P+1)) >= 1, name="mandatory_3")
model.addConstr(gp.quicksum(x[p, 6] + y[p, 6] for p in range(1, P+1)) >= 1, name="mandatory_6")

# 5.14 Extra incentive projects only if unlocked
for p in range(1, P+1):
    for j in J_extra:
        model.addConstr(y[p, j] <= w[p], name=f"extra_unlocked_{p}_{j}")

# ====================
# 6. SOLVE
# ====================
model.optimize()

# ====================
# 7. PRINT RESULTS
# ====================
if model.status == GRB.OPTIMAL:
    print(f"Optimal total profit: {model.objVal:.2f} million yuan")
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found.")
    print(f"FinalAnswer=【0】")