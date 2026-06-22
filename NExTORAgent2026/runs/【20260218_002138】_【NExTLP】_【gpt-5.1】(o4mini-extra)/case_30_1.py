import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters and data
# =========================

# Given parameter list (used exactly as provided)
established_year = 1985
top_three_positions = 3
num_players = 15
age_range = (18, 25)
avg_training_years = 8
record_period_months = 12
selection_pool_size = 15
num_events = 6
events = ['Horizontal Bar', 'Parallel Bars', 'Vaulting Horse',
          'Pommel Horse', 'Rings', 'Floor Exercise']
max_players_per_event = 6
movements_per_event = 2
score_range = (0, 18)
min_events_per_contestant = 3
min_allround_contestants = 2
bonus_threshold = 2
bonus_per_additional = 2
player_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
event_indices = [1, 2, 3, 4, 5, 6]
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

# =========================
# 2. Create model
# =========================
model = gp.Model("Hongwei_Gymnastics_Team_Selection")

# =========================
# 3. Decision variables
# =========================

# z_i: 1 if player i is selected to the team
z = model.addVars(player_indices, vtype=GRB.BINARY, name="z")

# x_{i,j}: 1 if player i competes in event j
x = model.addVars(player_indices, event_indices, vtype=GRB.BINARY, name="x")

# v_i: 1 if player i participates in all 6 events (all-around)
v = model.addVars(player_indices, vtype=GRB.BINARY, name="v")

# =========================
# 4. Objective function
# =========================
# Z = sum_{i,j} S_ij * x_ij + bonus_per_additional * (sum_i v_i - bonus_threshold)

score_term = gp.quicksum(
    score_matrix[i - 1][j - 1] * x[i, j]
    for i in player_indices
    for j in event_indices
)

bonus_term = bonus_per_additional * (
    gp.quicksum(v[i] for i in player_indices) - bonus_threshold
)

model.setObjective(score_term + bonus_term, GRB.MAXIMIZE)

# =========================
# 5. Constraints
# =========================

# Event capacity: sum_i x_{i,j} <= max_players_per_event  for all j
for j in event_indices:
    model.addConstr(
        gp.quicksum(x[i, j] for i in player_indices) <= max_players_per_event,
        name=f"EventCapacity_{j}"
    )

# Minimum events per player: sum_j x_{i,j} >= min_events_per_contestant * z_i  for all i
for i in player_indices:
    model.addConstr(
        gp.quicksum(x[i, j] for j in event_indices) >= min_events_per_contestant * z[i],
        name=f"MinEventsPerPlayer_{i}"
    )

# Link assignment to selection: x_{i,j} <= z_i  for all i,j
for i in player_indices:
    for j in event_indices:
        model.addConstr(
            x[i, j] <= z[i],
            name=f"AssignOnlyIfSelected_{i}_{j}"
        )

# All-around headcount: sum_i v_i >= min_allround_contestants
model.addConstr(
    gp.quicksum(v[i] for i in player_indices) >= min_allround_contestants,
    name="MinAllAround"
)

# All-around definition (lower bound): sum_j x_{i,j} >= num_events * v_i  for all i
for i in player_indices:
    model.addConstr(
        gp.quicksum(x[i, j] for j in event_indices) >= num_events * v[i],
        name=f"AllAroundLower_{i}"
    )

# All-around definition (upper bound): sum_j x_{i,j} <= (num_events - 1) + v_i for all i
for i in player_indices:
    model.addConstr(
        gp.quicksum(x[i, j] for j in event_indices) <= (num_events - 1) + v[i],
        name=f"AllAroundUpper_{i}"
    )

# =========================
# 6. Optimize
# =========================
model.optimize()

# =========================
# 7. Print results
# =========================

if model.Status == GRB.OPTIMAL:
    print(f"Optimal objective value (best expected team score) = {model.ObjVal:.4f}")
    print("\nSelected players (z_i = 1):")
    for i in player_indices:
        if z[i].X > 0.5:
            print(f" Player {i} selected.")

    print("\nAssignments x_{i,j} (player i in event j):")
    for i in player_indices:
        for j in event_indices:
            if x[i, j].X > 0.5:
                print(f" Player {i} competes in event {j} ({events[j-1]}).")

    print("\nAll-around players (v_i = 1):")
    for i in player_indices:
        if v[i].X > 0.5:
            print(f" Player {i} is all-around (all 6 events).")

    # Final answer is the best expected team score (objective value)
    final_answer_value = model.ObjVal
else:
    # In case of non-optimal termination, set a fallback value
    final_answer_value = float('nan')

# Required final output statement
print(f"FinalAnswer=【{final_answer_value}】")