import gurobipy as gp
from gurobipy import GRB

# Define the score matrix based on the problem description
# Rows: Athletes 1 to 15
# Cols: Events 1 to 6
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

# Parameters
num_players = 15
num_events = 6
min_events_per_contestant = 3
max_players_per_event = 6
min_allround_contestants = 2
bonus_per_additional = 2
bonus_threshold = 2

# Initialize Model
model = gp.Model("GymnasticsTeamOptimization")

# Decision Variables
# x[i, j] = 1 if player i competes in event j
x = {}
for i in range(num_players):
    for j in range(num_events):
        x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

# z[i] = 1 if player i is selected for the team
z = {}
for i in range(num_players):
    z[i] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}")

# v[i] = 1 if player i participates in all 6 events (all-around)
v = {}
for i in range(num_players):
    v[i] = model.addVar(vtype=GRB.BINARY, name=f"v_{i}")

# Update model to integrate new variables
model.update()

# Objective Function
# Total Score = Sum of individual event scores + Bonus
# Bonus = 2 points for each all-around competitor exceeding 2
# Formula: Bonus = 2 * (sum(v_i) - 2)
# Objective = Sum(S_ij * x_ij) + 2 * Sum(v_i) - 4
raw_score = gp.quicksum(score_matrix[i][j] * x[i, j] for i in range(num_players) for j in range(num_events))
bonus_score = bonus_per_additional * (gp.quicksum(v[i] for i in range(num_players)) - bonus_threshold)

model.setObjective(raw_score + bonus_score, GRB.MAXIMIZE)

# Constraints

# 1. Event Capacity: At most 6 players per event
for j in range(num_events):
    model.addConstr(gp.quicksum(x[i, j] for i in range(num_players)) <= max_players_per_event, name=f"Capacity_Event_{j}")

# 2. Link Assignment to Selection: x_ij <= z_i
for i in range(num_players):
    for j in range(num_events):
        model.addConstr(x[i, j] <= z[i], name=f"Link_x_z_{i}_{j}")

# 3. Minimum Events per Selected Player: Each participant must do at least 3 events
for i in range(num_players):
    model.addConstr(gp.quicksum(x[i, j] for j in range(num_events)) >= min_events_per_contestant * z[i], name=f"MinEvents_{i}")

# 4. All-Around Logic (Indicator Constraints)
# v[i] = 1 <-> sum(x_ij) = 6
# Implemented as: v[i]=1 -> sum(x_ij) >= 6 (since max is 6, implies =6)
#                 v[i]=0 -> sum(x_ij) <= 5
for i in range(num_players):
    # If v[i] is 1, player must participate in >= 6 events
    model.addGenConstrIndicator(v[i], 1, gp.quicksum(x[i, j] for j in range(num_events)) >= num_events, name=f"Ind_AllAround_Yes_{i}")
    # If v[i] is 0, player must participate in <= 5 events
    model.addGenConstrIndicator(v[i], 0, gp.quicksum(x[i, j] for j in range(num_events)) <= num_events - 1, name=f"Ind_AllAround_No_{i}")

# 5. All-Around Headcount: At least 2 people must participate in all 6 events
model.addConstr(gp.quicksum(v[i] for i in range(num_players)) >= min_allround_contestants, name="MinAllAroundCount")

# Solve the model
model.optimize()

# Output the result
if model.status == GRB.OPTIMAL:
    print(f"Optimal Objective Value: {model.objVal}")
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found.")