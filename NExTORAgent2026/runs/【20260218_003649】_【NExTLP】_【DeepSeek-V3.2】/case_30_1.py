import gurobipy as gp
from gurobipy import GRB

# Parameters from the provided list
num_players = 15
num_events = 6
max_players_per_event = 6
min_events_per_contestant = 3
min_allround_contestants = 2
bonus_threshold = 2
bonus_per_additional = 2

player_indices = list(range(num_players))  # 0 to 14 for Python indexing
event_indices = list(range(num_events))    # 0 to 5 for Python indexing

# Score matrix S_ij (player i, event j)
S = [
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

# Create model
model = gp.Model("GymnasticsTeamSelection")

# Decision variables
z = model.addVars(num_players, vtype=GRB.BINARY, name="z")
x = model.addVars(num_players, num_events, vtype=GRB.BINARY, name="x")
v = model.addVars(num_players, vtype=GRB.BINARY, name="v")

# Objective function: maximize sum of scores plus bonus for extra all-around participants
score_sum = gp.quicksum(S[i][j] * x[i, j] for i in player_indices for j in event_indices)
bonus = bonus_per_additional * (gp.quicksum(v[i] for i in player_indices) - bonus_threshold)
model.setObjective(score_sum + bonus, GRB.MAXIMIZE)

# Constraints
# 1. Event capacity: at most 6 players per event
for j in event_indices:
    model.addConstr(gp.quicksum(x[i, j] for i in player_indices) <= max_players_per_event, 
                    f"EventCapacity_{j}")

# 2. Minimum events per selected player (at least 3)
for i in player_indices:
    model.addConstr(gp.quicksum(x[i, j] for j in event_indices) >= min_events_per_contestant * z[i],
                    f"MinEvents_{i}")

# 3. Assignment only if selected
for i in player_indices:
    for j in event_indices:
        model.addConstr(x[i, j] <= z[i], f"AssignIfSelected_{i}_{j}")

# 4. At least 2 all-around participants
model.addConstr(gp.quicksum(v[i] for i in player_indices) >= min_allround_contestants,
                "MinAllAround")

# 5. All-around definition using addGenConstrIndicator
# If v_i = 1, then sum_j x_ij >= 6
# If v_i = 0, then sum_j x_ij <= 5
for i in player_indices:
    # v_i = 1 -> sum_j x_ij >= 6
    model.addGenConstrIndicator(v[i], 1, 
                                gp.quicksum(x[i, j] for j in event_indices) >= 6,
                                name=f"AllAroundDef1_{i}")
    # v_i = 0 -> sum_j x_ij <= 5
    model.addGenConstrIndicator(v[i], 0,
                                gp.quicksum(x[i, j] for j in event_indices) <= 5,
                                name=f"AllAroundDef0_{i}")

# Solve the model
model.optimize()

# Check if optimal solution found
if model.status == GRB.OPTIMAL:
    # Calculate the objective value
    obj_val = model.objVal
    
    # Print results
    print("Optimal solution found.")
    print(f"Best expected team score: {obj_val:.2f}")
    
    # Count selected players
    selected_players = [i for i in player_indices if z[i].X > 0.5]
    print(f"Number of selected players: {len(selected_players)}")
    
    # Count all-around participants
    all_around_count = sum(1 for i in player_indices if v[i].X > 0.5)
    print(f"Number of all-around participants: {all_around_count}")
    
    # Print event assignments
    print("\nEvent assignments (1 = assigned):")
    print("Player | " + " | ".join([f"E{j+1}" for j in event_indices]) + " | All-Around")
    for i in player_indices:
        if z[i].X > 0.5:
            assignments = [int(round(x[i, j].X)) for j in event_indices]
            all_around = int(round(v[i].X))
            print(f"{i+1:6} | " + " | ".join([f"{a:3}" for a in assignments]) + f" | {all_around:10}")
    
    # Print total scores per event
    print("\nTotal scores per event:")
    for j in event_indices:
        event_score = sum(S[i][j] * x[i, j].X for i in player_indices)
        print(f"Event {j+1}: {event_score:.2f}")
    
    # Calculate the base score (without bonus)
    base_score = sum(S[i][j] * x[i, j].X for i in player_indices for j in event_indices)
    bonus_points = bonus_per_additional * (all_around_count - bonus_threshold)
    print(f"\nBase score: {base_score:.2f}")
    print(f"Bonus points: {bonus_points:.2f}")
    print(f"Total score: {obj_val:.2f}")
    
    # Output the final answer as required
    print(f"FinalAnswer=【{obj_val:.2f}】")
else:
    print("No optimal solution found.")
    print(f"FinalAnswer=【0.00】")