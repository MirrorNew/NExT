import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters (from Parameters List)
# =========================

num_gymnasts = 15
num_events = 6

scores_S_ij = [
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

max_gymnasts_per_event = 6
min_events_per_gymnast = 3
min_all_around_gymnists = 2  # used only in the objective structure comment
bonus_per_additional_all_around = 2

# For clarity we keep exactly the structure given:
# objective_function_structure:
# 'maximize sum_{i=1..15} sum_{j=1..6} S[i][j]*x[i,j] + 2*(sum_{i=1..15} y[i] - 2)'
# constraint_event_capacity:
# 'for each event j: sum_{i=1..15} x[i,j] <= 6'
# constraint_min_events_per_gymnast:
# 'for each gymnast i: sum_{j=1..6} x[i,j] >= 3'
# constraint_all_around_link_1:
# 'for each i,j: x[i,j] >= y[i]'
# constraint_all_around_link_2:
# 'for each i: sum_{j=1..6} x[i,j] <= 6*y[i]'
# constraint_min_all_around:
# 'sum_{i=1..15} y[i] >= 2'
optimal_team_score_given = 562.5  # target optimal value from Parameters List (for checking)

# =========================
# 2. Create Model
# =========================

model = gp.Model("Hongwei_Gymnastics_Team_Selection")

# =========================
# 3. Decision Variables
# =========================

# x[i,j] in {0,1}
x = model.addVars(
    num_gymnasts,
    num_events,
    vtype=GRB.BINARY,
    name="x"
)

# y[i] in {0,1}
y = model.addVars(
    num_gymnasts,
    vtype=GRB.BINARY,
    name="y"
)

# =========================
# 4. Objective Function
# =========================

# maximize sum_{i=1..15} sum_{j=1..6} S[i][j]*x[i,j] + 2*(sum_{i=1..15} y[i] - 2)
model.setObjective(
    gp.quicksum(scores_S_ij[i][j] * x[i, j]
                for i in range(num_gymnasts)
                for j in range(num_events))
    + bonus_per_additional_all_around * (
        gp.quicksum(y[i] for i in range(num_gymnasts)) - 2
    ),
    GRB.MAXIMIZE
)

# =========================
# 5. Constraints
# =========================

# constraint_event_capacity:
# for each event j: sum_{i=1..15} x[i,j] <= 6
for j in range(num_events):
    model.addConstr(
        gp.quicksum(x[i, j] for i in range(num_gymnasts)) <= max_gymnasts_per_event,
        name=f"event_capacity_e{j+1}"
    )

# constraint_min_events_per_gymnast:
# for each gymnast i: sum_{j=1..6} x[i,j] >= 3
for i in range(num_gymnasts):
    model.addConstr(
        gp.quicksum(x[i, j] for j in range(num_events)) >= min_events_per_gymnast,
        name=f"min_events_g{i+1}"
    )

# constraint_all_around_link_1:
# for each i,j: x[i,j] >= y[i]
for i in range(num_gymnasts):
    for j in range(num_events):
        model.addConstr(
            x[i, j] >= y[i],
            name=f"all_around_link1_g{i+1}_e{j+1}"
        )

# constraint_all_around_link_2:
# for each i: sum_{j=1..6} x[i,j] <= 6*y[i]
for i in range(num_gymnasts):
    model.addConstr(
        gp.quicksum(x[i, j] for j in range(num_events)) <= num_events * y[i],
        name=f"all_around_link2_g{i+1}"
    )

# constraint_min_all_around:
# sum_{i=1..15} y[i] >= 2
model.addConstr(
    gp.quicksum(y[i] for i in range(num_gymnasts)) >= 2,
    name="min_all_around"
)

# =========================
# 6. Optimize
# =========================

model.optimize()

# =========================
# 7. Retrieve and Print Results
# =========================

if model.Status == GRB.OPTIMAL:
    optimal_value = model.ObjVal

    print("Optimal team score from model:", optimal_value)
    print("Optimal team score (given in Parameters List):", optimal_team_score_given)
    print("\nSelected gymnasts and their events (1 = selected):")
    for i in range(num_gymnasts):
        row = [int(round(x[i, j].X)) for j in range(num_events)]
        print(f"Gymnast {i+1}: {row}  | all-around y[{i+1}] = {int(round(y[i].X))}")

    # Explicit final answer format
    print(f"FinalAnswer=【{optimal_value}】")
else:
    print("No optimal solution found.")
    print("FinalAnswer=【None】")