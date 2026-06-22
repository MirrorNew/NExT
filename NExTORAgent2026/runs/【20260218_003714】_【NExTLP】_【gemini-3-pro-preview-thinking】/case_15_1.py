import gurobipy as gp
from gurobipy import GRB

# 1. Initialize Model
model = gp.Model("Shengrui_Supply_Chain_Investment")

# 2. Parameters
# Periods
phases = list(range(1, 13))
n_phases = 12

# Projects
projects = list(range(1, 7))
# Project Data: ID -> (Cost, Profit, Manpower, Type)
# Type: 0 for R&D (1,2,3), 1 for Implementation (4,5,6)
project_data = {
    1: {'cost': 10, 'profit': 8,  'hr': 4, 'type': 'RD'},
    2: {'cost': 15, 'profit': 12, 'hr': 6, 'type': 'RD'},
    3: {'cost': 8,  'profit': 7,  'hr': 3, 'type': 'RD'},
    4: {'cost': 20, 'profit': 15, 'hr': 8, 'type': 'Imp'},
    5: {'cost': 11, 'profit': 9,  'hr': 5, 'type': 'Imp'},
    6: {'cost': 9,  'profit': 8,  'hr': 4, 'type': 'Imp'}
}

budget_limit = 40
hr_limit = 18

# 3. Decision Variables
# x[p, j]: Normal selection of project j in period p
x = model.addVars(phases, projects, vtype=GRB.BINARY, name="x")

# y[p, j]: Incentive selection of project j in period p (Only for Imp projects 4,5,6)
imp_projects = [4, 5, 6]
y = model.addVars(phases, imp_projects, vtype=GRB.BINARY, name="y")

# w[p]: Indicator if incentives are unlocked/available in period p
w = model.addVars(phases, vtype=GRB.BINARY, name="w")

# z variables: Cumulative thresholds met by end of period p
# z1 for P1>=3, z2 for P2>=2, z3 for P3>=1
z1 = model.addVars(phases, vtype=GRB.BINARY, name="z1")
z2 = model.addVars(phases, vtype=GRB.BINARY, name="z2")
z3 = model.addVars(phases, vtype=GRB.BINARY, name="z3")

# alpha[p]: Synergy trigger (P3 + P6 selected in p)
alpha = model.addVars(phases, vtype=GRB.BINARY, name="alpha")

# s[p]: Synergy utilization (P3 cost halved in p)
# Defined for p=2..12 mostly, but can define for all and constrain s[1]=0
s = model.addVars(phases, vtype=GRB.BINARY, name="s")

# 4. Objective Function
# Maximize total profit (Normal + Incentive)
# Profit is same for y, cost is handled in constraints
obj_expr = gp.LinExpr()
for p in phases:
    for j in projects:
        obj_expr += project_data[j]['profit'] * x[p, j]
    for j in imp_projects:
        obj_expr += project_data[j]['profit'] * y[p, j]

model.setObjective(obj_expr, GRB.MAXIMIZE)

# 5. Constraints

# 5.1 Budget & Manpower per period
for p in phases:
    # Budget: Normal Cost + 0.7 * Extra Cost - Synergy Discount <= 40
    # Synergy discount: Half of P3 cost (8/2 = 4) if s[p] is active
    cost_expr = gp.LinExpr()
    hr_expr = gp.LinExpr()
    
    # Normal projects
    for j in projects:
        cost_expr += project_data[j]['cost'] * x[p, j]
        hr_expr += project_data[j]['hr'] * x[p, j]
        
    # Extra projects (0.7 cost, 0 manpower)
    for j in imp_projects:
        cost_expr += 0.7 * project_data[j]['cost'] * y[p, j]
        # Manpower for y is 0 (or company sends additional people), so hr_expr is untouched by y
        
    # Synergy deduction
    cost_expr -= 4 * s[p]
    
    model.addConstr(cost_expr <= budget_limit, name=f"Budget_{p}")
    model.addConstr(hr_expr <= hr_limit, name=f"Manpower_{p}")

# 5.2 Dependencies & Logic
for p in phases:
    # P2 needs P1 (x_p2 <= x_p1) - Applies to normal selection
    model.addConstr(x[p, 2] <= x[p, 1], name=f"Dep_P2_P1_{p}")
    
    # P4 and P5 Exclusivity (cannot select both in any form)
    # (x4 + y4) + (x5 + y5) <= 1
    model.addConstr(
        (x[p, 4] + y[p, 4]) + (x[p, 5] + y[p, 5]) <= 1, 
        name=f"Excl_P4_P5_{p}"
    )
    
    # A project cannot be both normal and extra in the same period
    for j in imp_projects:
        model.addConstr(x[p, j] + y[p, j] <= 1, name=f"SingleMode_{p}_{j}")

# 5.3 Selection Counts per Period
for p in phases:
    # Normal projects max 4
    model.addConstr(gp.quicksum(x[p, j] for j in projects) <= 4, name=f"MaxNormal_{p}")
    
    # R&D projects (1,2,3) count between 1 and 2
    rd_sum = gp.quicksum(x[p, j] for j in [1, 2, 3])
    model.addConstr(rd_sum >= 1, name=f"MinRD_{p}")
    model.addConstr(rd_sum <= 2, name=f"MaxRD_{p}")
    
    # Extra incentive projects: Max 1, and only if unlocked (w[p]=1)
    extra_sum = gp.quicksum(y[p, j] for j in imp_projects)
    model.addConstr(extra_sum <= w[p], name=f"ExtraLimit_{p}")

# 5.4 Synergy Logic (P3 & P6)
for p in phases:
    # Alpha definition: alpha[p] = 1 <-> x[p,3]=1 AND x[p,6]=1
    # Linearization using indicators or standard inequalities
    model.addConstr(alpha[p] <= x[p, 3])
    model.addConstr(alpha[p] <= x[p, 6])
    model.addConstr(alpha[p] >= x[p, 3] + x[p, 6] - 1)
    
    # s[p] definition: s[p] = 1 <-> alpha[p-1]=1 AND x[p,3]=1
    if p == 1:
        model.addConstr(s[p] == 0) # No synergy in first period
    else:
        model.addConstr(s[p] <= alpha[p-1])
        model.addConstr(s[p] <= x[p, 3])
        # We need s[p] to trigger if possible to save budget (which helps objective indirectly or feasibility)
        # Strictly defining the relationship:
        model.addConstr(s[p] >= alpha[p-1] + x[p, 3] - 1)

# 5.5 Incentive Unlocking Logic
# Cumulative counts
for p in phases:
    # Calculate cumulative sums up to p
    cum_p1 = gp.quicksum(x[t, 1] for t in range(1, p + 1))
    cum_p2 = gp.quicksum(x[t, 2] for t in range(1, p + 1))
    cum_p3 = gp.quicksum(x[t, 3] for t in range(1, p + 1))
    
    # Indicator constraints for z variables
    # z1[p] = 1 <-> cum_p1 >= 3
    model.addGenConstrIndicator(z1[p], 1, cum_p1 >= 3)
    model.addGenConstrIndicator(z1[p], 0, cum_p1 <= 2)
    
    # z2[p] = 1 <-> cum_p2 >= 2
    model.addGenConstrIndicator(z2[p], 1, cum_p2 >= 2)
    model.addGenConstrIndicator(z2[p], 0, cum_p2 <= 1)
    
    # z3[p] = 1 <-> cum_p3 >= 1
    model.addGenConstrIndicator(z3[p], 1, cum_p3 >= 1)
    model.addGenConstrIndicator(z3[p], 0, cum_p3 <= 0)
    
    # w[p] availability: w[p] = 1 if z variables of p-1 are all 1
    if p == 1:
        model.addConstr(w[p] == 0)
    else:
        # w[p] <= z[p-1]
        model.addConstr(w[p] <= z1[p-1])
        model.addConstr(w[p] <= z2[p-1])
        model.addConstr(w[p] <= z3[p-1])
        # Since w allows extra profit (y vars), objective will push w to 1 if possible.
        # But for strict correctness:
        model.addConstr(w[p] >= z1[p-1] + z2[p-1] + z3[p-1] - 2)

# 5.6 Annual Goals
# Each project at least once
for j in projects:
    total_sel = gp.quicksum(x[p, j] for p in phases)
    if j in imp_projects:
        total_sel += gp.quicksum(y[p, j] for p in phases)
    model.addConstr(total_sel >= 1, name=f"AnnualMin_{j}")

# 6. Solve and Output
model.optimize()

if model.Status == GRB.OPTIMAL:
    print(f"Optimal Objective Value: {model.ObjVal}")
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("No optimal solution found.")