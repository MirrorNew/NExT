import gurobipy as gp

# Chenxi Reservoir optimization with Gurobi
# Note: all numeric values strictly come from the given Parameters List

# 1. Create model
model = gp.Model("Chenxi_Reservoir_Scheduling")

# 2. Parameters (from Parameters List; do not modify or invent)
initial_storage_10k_m3 = 100
num_periods = 4
storage_lower_bound_10k_m3 = 0
storage_upper_bound_10k_m3 = 200
final_storage_min_10k_m3 = 20
release_lower_bound_10k_m3 = 0
release_upper_bound_10k_m3 = 100
units = "10000 m3"

Table_1_periods = [1, 2, 3, 4]
Table_1_inflow_10k_m3 = [80, 50, 20, 0]
Table_1_max_supply_region_A_10k_m3 = [20, 40, 55, 50]
Table_1_max_supply_region_B_10k_m3 = [10, 30, 40, 40]
Table_1_min_supply_region_A_10k_m3 = [10, 20, 30, 20]
Table_1_min_supply_region_B_10k_m3 = [8, 20, 30, 34]
Table_1_power_generation_rate = [0.31, 1.55, 2.05, 0.65]
power_weight_periods_1_to_4 = [2, 2, 1, 1]
total_initial_plus_inflow_10k_m3 = 250
max_total_release_given_final_storage_min_10k_m3 = 230

# 3. Index sets
T = Table_1_periods  # [1, 2, 3, 4]

# 4. Decision variables
# Total release in each period
R = model.addVars(
    T,
    lb=release_lower_bound_10k_m3,
    ub=release_upper_bound_10k_m3,
    vtype=gp.GRB.CONTINUOUS,
    name="R",
)

# Storage at end of each period
S = model.addVars(
    T,
    lb=storage_lower_bound_10k_m3,
    ub=storage_upper_bound_10k_m3,
    vtype=gp.GRB.CONTINUOUS,
    name="S",
)

# Supplies to region A and B
A = model.addVars(
    T,
    lb=0.0,  # will also be bounded by min/max constraints from the parameters
    vtype=gp.GRB.CONTINUOUS,
    name="A",
)
B = model.addVars(
    T,
    lb=0.0,
    vtype=gp.GRB.CONTINUOUS,
    name="B",
)

# 5. Constraints

# 5.1 Water balance
# S_1 = S_0 + I_1 - R_1
model.addConstr(
    S[1] == initial_storage_10k_m3 + Table_1_inflow_10k_m3[0] - R[1],
    name="WaterBalance_1",
)

# For t = 2,3,4: S_t = S_{t-1} + I_t - R_t
for idx, t in enumerate(T[1:], start=1):  # idx:1..3 for t:2..4
    inflow_t = Table_1_inflow_10k_m3[idx]
    prev_t = T[idx - 1]
    model.addConstr(
        S[t] == S[prev_t] + inflow_t - R[t],
        name=f"WaterBalance_{t}",
    )

# 5.2 Terminal storage requirement: S_4 ≥ 20
model.addConstr(
    S[4] >= final_storage_min_10k_m3,
    name="TerminalStorage",
)

# 5.3 Supply–release linkage: A_t + B_t = R_t
for t in T:
    model.addConstr(
        A[t] + B[t] == R[t],
        name=f"SupplyReleaseLinkage_{t}",
    )

# 5.4 Region A supply limits: A_t^min ≤ A_t ≤ A_t^max
for idx, t in enumerate(T):
    A_min = Table_1_min_supply_region_A_10k_m3[idx]
    A_max = Table_1_max_supply_region_A_10k_m3[idx]
    model.addConstr(A[t] >= A_min, name=f"A_Min_{t}")
    model.addConstr(A[t] <= A_max, name=f"A_Max_{t}")

# 5.5 Region B supply limits: B_t^min ≤ B_t ≤ B_t^max
for idx, t in enumerate(T):
    B_min = Table_1_min_supply_region_B_10k_m3[idx]
    B_max = Table_1_max_supply_region_B_10k_m3[idx]
    model.addConstr(B[t] >= B_min, name=f"B_Min_{t}")
    model.addConstr(B[t] <= B_max, name=f"B_Max_{t}")

# 5.6 (Optional but implied by “no discard, no artificial increase”)
# Total water used (total release + final storage) equals initial + inflow.
# Since S_4 = S_0 + sum(I_t) − sum(R_t), this is already enforced by
# the water-balance equations and variable bounds, so no extra constraint needed.


# 6. Objective function
# As requested in the validated mathematical model:
# Max Z = 0.31 R_1 + 1.55 R_2 + 2.05 R_3 + 0.65 R_4
obj_expr = gp.LinExpr()
for idx, t in enumerate(T):
    rate_t = Table_1_power_generation_rate[idx]
    obj_expr += rate_t * R[t]

model.setObjective(obj_expr, gp.GRB.MAXIMIZE)

# 7. Solve model
model.optimize()

# 8. Print results + FinalAnswer as required
if model.status == gp.GRB.OPTIMAL:
    print("\nOptimal solution found.")
    print(f"Objective value (total power generation, in arbitrary units): {model.objVal:.4f}")
    print(f"Units of release/storage/supply: {units}")

    print("\nPeriod |   R_t  |   S_t  |   A_t  |   B_t  (all in 10,000 m3)")
    for t in T:
        print(
            f"{t:>6} | "
            f"{R[t].X:6.2f} | "
            f"{S[t].X:6.2f} | "
            f"{A[t].X:6.2f} | "
            f"{B[t].X:6.2f}"
        )

    # The question asks: "please give the calculated total power generation."
    # According to the validated objective, that is model.objVal.
    FinalAnswer_value = model.objVal
else:
    print(f"Optimization ended with status {model.status} (not optimal).")
    FinalAnswer_value = float('nan')

# Mandatory final line: print FinalAnswer in the required format
print(f"FinalAnswer=【{FinalAnswer_value}】")