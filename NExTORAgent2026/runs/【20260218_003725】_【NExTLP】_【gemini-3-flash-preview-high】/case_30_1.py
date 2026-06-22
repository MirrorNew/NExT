import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs
num_players = 15
num_events = 6
events = ['Horizontal Bar', 'Parallel Bars', 'Vaulting Horse', 'Pommel Horse', 'Rings', 'Floor Exercise']
score_matrix = [
    [16.0, 14.9, 10.5, 8.4, 11.6, 10.3],
    [15.2, 8.9, 11.2, 12.6, 16.8, 11.6],
    [8.7, 14.8, 13.0, 8.3, 16.8, 17.8],
    [15.5, 16.7, 9.0, 14.5, 16.7, 13.9],
    [11.1, 6.3, 10.6, 12.9, 16.9, 17.6],
    [11.2, 16.2, 8.4, 15.5, 12.1, 5.2],
    [14.4, 10.2, 15.7, 13.7, 5.0, 11.4],
    [16.3, 8.2, 9.2, 16.3, 7.5, 12.4],
    [8.1, 17.6, 15.4, 10.8, 6.0, 9.2],
    [11.6, 17.1, 6.4, 12.2, 14.2, 12.1],
    [15.6, 12.0, 17.5, 12.8, 12.6, 10.8],
    [12.8, 10.0, 12.5, 8.8, 7.5, 7.4],
    [13.0, 13.5, 11.2, 6.2, 14.8, 16.4],
    [17.0, 16.0, 16.7, 17.0, 12.0, 10.1],
    [14.2, 8.6, 15.6, 16.0, 16.6, 12.7]
]

# 2. Create the model
model = gp.Model("GymnasticsTeamSelection")

# 3. Create decision variables
# x[i, j] = 1 if athlete i competes in event j, 0 otherwise
x = model.addVars(num_players, num_events, vtype=GRB.BINARY, name="x")
# z[i] = 1 if athlete i is selected for the team, 0 otherwise
z = model.addVars(num_players, vtype=GRB.BINARY, name="z")
# v[i] = 1 if athlete i is an all-around competitor (competes in all 6 events), 0 otherwise
v = model.addVars(num_players, vtype=GRB.BINARY, name="v")

# 4. Set up the objective function
# Objective: Maximize sum(Sij * xij) + 2 * (sum(vi) - 2)
score_sum = gp.quicksum(score_matrix[i][j] * x[i, j] for i in range(num_players) for j in range(num_events))
num_all_around = gp.quicksum(v[i] for i in range(num_players))
# Bonus for extra all-around contestants beyond the first 2
bonus = 2 * (num_all_around - 2)
model.setObjective(score_sum + bonus, GRB.MAXIMIZE)

# 5. Add all constraints
# Event capacity: At most 6 players in each event
for j in range(num_events):
    model.addConstr(gp.quicksum(x[i, j] for i in range(num_players)) <= 6, name=f"EventCapacity_{j}")

# Link assignment to selection: x[i,j] <= z[i]
for i in range(num_players):
    for j in range(num_events):
        model.addConstr(x[i, j] <= z[i], name=f"Link_x_z_{i}_{j}")

# Minimum events per contestant: Each selected player must compete in at least 3 events
for i in range(num_players):
    model.addGenConstrIndicator(z[i], 1, gp.quicksum(x[i, j] for j in range(num_events)) >= 3)
    model.addGenConstrIndicator(z[i], 0, gp.quicksum(x[i, j] for j in range(num_events)) == 0)

# All-around headcount: At least 2 people must participate in all 6 events
model.addConstr(gp.quicksum(v[i] for i in range(num_players)) >= 2, name="MinAllAround")

# All-around definition: v[i] is 1 iff all 6 events are selected for athlete i
for i in range(num_players):
    model.addGenConstrIndicator(v[i], 1, gp.quicksum(x[i, j] for j in range(num_events)) == 6)
    model.addGenConstrIndicator(v[i], 0, gp.quicksum(x[i, j] for j in range(num_events)) <= 5)

# 6. Solve the model and print results
model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"Optimal Expected Score: {model.objVal}")
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found.")