import gurobipy as gp
from gurobipy import GRB

# Create the model
model = gp.Model("ShengruiSupplyChainTechnology")

# Parameters
num_phases = 12
num_projects = 6
budget_limit_per_period = 40
hr_limit_per_period = 18

# Project Info: Project Name, Category, Cost (M yuan), Profit (M yuan), Manpower Requirements (People)
# R&D: Projects 1, 2, 3 (Indices 0, 1, 2)
# Implementation: Projects 4, 5, 6 (Indices 3, 4, 5)
costs = [10, 15, 8, 20, 11, 9]
profits = [8, 12, 7, 15, 9, 8]
manpower = [4, 6, 3, 8, 5, 4]

# Decision Variables
# x[p, j] = 1 if project j is selected in period p under normal rules
x = model.addVars(num_phases, num_projects, vtype=GRB.BINARY, name="x")
# y[p, j] = 1 if implementation project j is selected as an extra incentive project in period p
# Projects 4, 5, 6 correspond to indices 3, 4, 5
y = model.addVars(num_phases, range(3, 6), vtype=GRB.BINARY, name="y")

# Incentive related variables
# w[p] = 1 if incentives are unlocked by the start of period p (using conditions from period 0 to p-1)
w = model.addVars(num_phases, vtype=GRB.BINARY, name="w")
# z1, z2, z3: 1 if cumulative selection goals for P1, P2, P3 are met by the end of period p
z1 = model.addVars(num_phases, vtype=GRB.BINARY, name="z1")
z2 = model.addVars(num_phases, vtype=GRB.BINARY, name="z2")
z3 = model.addVars(num_phases, vtype=GRB.BINARY, name="z3")

# Synergy variables
# alpha[p] = 1 if both Project 3 and Project 6 are selected in period p
alpha = model.addVars(num_phases, vtype=GRB.BINARY, name="alpha")
# s[p] = 1 if cost of Project 3 is halved in period p (requires alpha in p-1 and P3 in p)
s = model.addVars(num_phases, vtype=GRB.BINARY, name="s")

# Objective Function: Maximize cumulative annual profit
total_profit = gp.quicksum(profits[j] * x[p, j] for p in range(num_phases) for j in range(num_projects)) + \
               gp.quicksum(profits[j] * y[p, j] for p in range(num_phases) for j in range(3, 6))
model.setObjective(total_profit, GRB.MAXIMIZE)

# Constraints
for p in range(num_phases):
    # Budget per period (Project 3 cost halved if s[p] is 1, Extra implementation cost 0.7 * cost)
    model.addConstr(
        gp.quicksum(costs[j] * x[p, j] for j in range(num_projects)) +
        gp.quicksum(0.7 * costs[j] * y[p, j] for j in range(3, 6)) -
        0.5 * costs[2] * s[p] <= budget_limit_per_period,
        name=f"Budget_Constraint_{p}"
    )
    
    # Manpower per period (Additional implementation projects do not count toward HR constraints)
    model.addConstr(
        gp.quicksum(manpower[j] * x[p, j] for j in range(num_projects)) <= hr_limit_per_period,
        name=f"Manpower_Constraint_{p}"
    )
    
    # Dependency: Project 2 (index 1) requires Project 1 (index 0) to be selected simultaneously
    model.addConstr(x[p, 1] <= x[p, 0], name=f"Dependency_{p}")
    
    # Exclusivity: Project 4 (index 3) and Project 5 (index 4) cannot be implemented together
    model.addConstr(x[p, 3] + y[p, 3] + x[p, 4] + y[p, 4] <= 1, name=f"Exclusivity_4_5_{p}")
    
    # Extra selection restriction for Project 6 (preventing selection as both normal and extra in same period)
    model.addConstr(x[p, 5] + y[p, 5] <= 1, name=f"Exclusivity_6_{p}")
    
    # Total project limit per period (Max 4 standard, and if w[p]=1, max 1 extra implementation)
    model.addConstr(gp.quicksum(x[p, j] for j in range(num_projects)) <= 4, name=f"ProjectLimitX_{p}")
    model.addConstr(gp.quicksum(y[p, j] for j in range(3, 6)) <= w[p], name=f"ProjectLimitY_{p}")
    # Overall selection limit including incentive bonus
    model.addConstr(gp.quicksum(x[p, j] for j in range(num_projects)) +
                    gp.quicksum(y[p, j] for j in range(3, 6)) <= 4 + w[p], name=f"TotalProjectLimit_{p}")
    
    # R&D Selection Restriction: 1 <= count of R&D projects (1, 2, 3) <= 2
    model.addConstr(gp.quicksum(x[p, j] for j in range(3)) >= 1, name=f"RDMin_{p}")
    model.addConstr(gp.quicksum(x[p, j] for j in range(3)) <= 2, name=f"RDMax_{p}")
    
    # Synergy trigger: alpha[p] is 1 if P3 (index 2) and P6 (index 5) are selected
    model.addGenConstrAnd(alpha[p], [x[p, 2], x[p, 5]], name=f"SynergyTrigger_{p}")

    # Define milestone z variables based on cumulative counts up to period p
    # Project 1: Selected 3 times
    model.addGenConstrIndicator(z1[p], 1, gp.quicksum(x[t, 0] for t in range(p+1)) >= 3)
    model.addGenConstrIndicator(z1[p], 0, gp.quicksum(x[t, 0] for t in range(p+1)) <= 2)
    # Project 2: Selected 2 times
    model.addGenConstrIndicator(z2[p], 1, gp.quicksum(x[t, 1] for t in range(p+1)) >= 2)
    model.addGenConstrIndicator(z2[p], 0, gp.quicksum(x[t, 1] for t in range(p+1)) <= 1)
    # Project 3: Selected 1 time
    model.addGenConstrIndicator(z3[p], 1, gp.quicksum(x[t, 2] for t in range(p+1)) >= 1)
    model.addGenConstrIndicator(z3[p], 0, gp.quicksum(x[t, 2] for t in range(p+1)) <= 0)

# Conditions for period 0 (cannot have prior-period dependencies)
model.addConstr(w[0] == 0)
model.addConstr(s[0] == 0)

# Incentives and synergy benefits logic for p > 0
for p in range(1, num_phases):
    # w[p] is unlocked if all milestone conditions (z1, z2, z3) were met by p-1
    model.addGenConstrAnd(w[p], [z1[p-1], z2[p-1], z3[p-1]], name=f"Unlock_Incentive_{p}")
    # Synergy benefit s[p] applies if triggered in p-1 and P3 selected in p
    model.addGenConstrAnd(s[p], [alpha[p-1], x[p, 2]], name=f"Apply_Synergy_Benefit_{p}")

# Annual goal: Each project selected at least once (Project 1-3 only x, Project 4-6 can be x or y)
model.addConstr(gp.quicksum(x[p, 0] for p in range(num_phases)) >= 3, name="Annual_P1_Req")
model.addConstr(gp.quicksum(x[p, 1] for p in range(num_phases)) >= 2, name="Annual_P2_Req")
model.addConstr(gp.quicksum(x[p, 2] for p in range(num_phases)) >= 1, name="Annual_P3_Req")
for j in range(3, 6):
    model.addConstr(gp.quicksum(x[p, j] + y[p, j] for p in range(num_phases)) >= 1, name=f"Annual_P{j+1}_Req")

# Solve
model.optimize()

# Output Results
if model.status == GRB.OPTIMAL:
    print(f"FinalAnswer=【{model.objVal}】")