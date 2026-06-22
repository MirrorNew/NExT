import gurobipy as gp
from gurobipy import GRB

# ==========================
# 1. Parameters and Data
# ==========================

# Parameters List values (must be used as given)
number_of_power_plants = 17
number_of_time_periods = 4
period_length_hours = 6
initial_output_before_period1 = 0
period_demands = [None, 11832.52, 14467.92, 16661.1, 15434.44]
transmission_loss_factor = 0
min_units_running = 8
initial_machine_state = 0

Table_1_Period_demand = {
    'I': 11832.52,
    'II': 14467.92,
    'III': 16661.1,
    'IV': 15434.44
}

Table_2_Power_plant_parameters = {
    'Biomass 1':      {'min_power': 78.75,  'max_power': 295.14, 'change_cost': 3.94, 'fixed_cost': 265.524, 'startup_cost': 0,   'shutdown_cost': 330},
    'Biomass 2':      {'min_power': 100.0,  'max_power': 300.0,  'change_cost': 4.1,  'fixed_cost': 270.0,   'startup_cost': 0,   'shutdown_cost': 340},
    'Combined Cycle 1': {'min_power': 742.5, 'max_power': 2883.9,'change_cost': 2.72, 'fixed_cost': 92.28,   'startup_cost': 500, 'shutdown_cost': 210},
    'Combined Cycle 2': {'min_power': 800.0, 'max_power': 3000.0,'change_cost': 2.85, 'fixed_cost': 95.0,    'startup_cost': 500, 'shutdown_cost': 215},
    'Combined Cycle 3': {'min_power': 700.0, 'max_power': 2800.0,'change_cost': 2.65, 'fixed_cost': 90.0,    'startup_cost': 500, 'shutdown_cost': 205},
    'Combined Cycle 4': {'min_power': 750.0, 'max_power': 2900.0,'change_cost': 2.8,  'fixed_cost': 93.0,    'startup_cost': 500, 'shutdown_cost': 208},
    'Wind Energy 1':  {'min_power': 246.0,  'max_power': 294.9,  'change_cost': 0.0,  'fixed_cost': 149.778, 'startup_cost': 0,   'shutdown_cost': 240},
    'Wind Energy 2':  {'min_power': 200.0,  'max_power': 350.0,  'change_cost': 0.0,  'fixed_cost': 160.0,   'startup_cost': 0,   'shutdown_cost': 250},
    'Wind Energy 3':  {'min_power': 220.0,  'max_power': 330.0,  'change_cost': 0.0,  'fixed_cost': 155.0,   'startup_cost': 0,   'shutdown_cost': 245},
    'Wind Energy 4':  {'min_power': 250.0,  'max_power': 340.0,  'change_cost': 0.0,  'fixed_cost': 162.0,   'startup_cost': 0,   'shutdown_cost': 255},
    'Hydropower 1':   {'min_power': 50.0,   'max_power': 500.0,  'change_cost': 1.5,  'fixed_cost': 60.0,    'startup_cost': 0,   'shutdown_cost': 100},
    'Hydropower 2':   {'min_power': 60.0,   'max_power': 600.0,  'change_cost': 1.8,  'fixed_cost': 65.0,    'startup_cost': 0,   'shutdown_cost': 120},
    'Hydropower3':    {'min_power': 55.0,   'max_power': 550.0,  'change_cost': 1.6,  'fixed_cost': 62.0,    'startup_cost': 0,   'shutdown_cost': 110},
    'Solar1':         {'min_power': 20.0,   'max_power': 400.0,  'change_cost': 0.0,  'fixed_cost': 120.0,   'startup_cost': 0,   'shutdown_cost': 150},
    'Solar2':         {'min_power': 20.0,   'max_power': 350.0,  'change_cost': 0.0,  'fixed_cost': 110.0,   'startup_cost': 0,   'shutdown_cost': 140},
    'Coal1':          {'min_power': 500.0,  'max_power': 2500.0, 'change_cost': 5.0,  'fixed_cost': 320.0,   'startup_cost': 500, 'shutdown_cost': 400},
    'Coal2':          {'min_power': 550.0,  'max_power': 2600.0, 'change_cost': 4.8,  'fixed_cost': 315.0,   'startup_cost': 480, 'shutdown_cost': 390}
}

# Build ordered list of plant names to index them 1..17
plant_names = list(Table_2_Power_plant_parameters.keys())
assert len(plant_names) == number_of_power_plants

# Helper: index sets
I = range(1, number_of_power_plants + 1)
T = range(1, number_of_time_periods + 1)

# Parameter arrays indexed by i
Pmin = {i: Table_2_Power_plant_parameters[plant_names[i-1]]['min_power'] for i in I}
Pmax = {i: Table_2_Power_plant_parameters[plant_names[i-1]]['max_power'] for i in I}
chg_cost = {i: Table_2_Power_plant_parameters[plant_names[i-1]]['change_cost'] for i in I}
fix_cost = {i: Table_2_Power_plant_parameters[plant_names[i-1]]['fixed_cost'] for i in I}
start_cost = {i: Table_2_Power_plant_parameters[plant_names[i-1]]['startup_cost'] for i in I}
shut_cost = {i: Table_2_Power_plant_parameters[plant_names[i-1]]['shutdown_cost'] for i in I}

# Demand per period (use given list with dummy at index 0)
D = {t: period_demands[t] for t in T}

# Combined-cycle units indices (from context mapping)
# 3: Combined Cycle 1, 4: Combined Cycle 2, 5: Combined Cycle 3, 6: Combined Cycle 4
combined_cycle_units = [3, 4, 5, 6]

# Initial conditions
y0 = {i: initial_machine_state for i in I}
p0 = {i: initial_output_before_period1 for i in I}

# ==========================
# 2. Create Model
# ==========================

model = gp.Model("Orissa_UnitCommitment_MinCost")

# ==========================
# 3. Decision Variables
# ==========================

# y[i,t]: on/off status
y = model.addVars(I, T, vtype=GRB.BINARY, name="y")

# p[i,t]: power output (MWh)
p = model.addVars(I, T, vtype=GRB.CONTINUOUS, lb=0.0, name="p")

# u[i,t]: startup indicator
u = model.addVars(I, T, vtype=GRB.BINARY, name="u")

# d[i,t]: shutdown indicator
d = model.addVars(I, T, vtype=GRB.BINARY, name="d")

# Delta+ and Delta- (change in output)
Delta_pos = model.addVars(I, T, vtype=GRB.CONTINUOUS, lb=0.0, name="Delta_pos")
Delta_neg = model.addVars(I, T, vtype=GRB.CONTINUOUS, lb=0.0, name="Delta_neg")

# ==========================
# 4. Constraints
# ==========================

# Generation limits: Pmin * y <= p <= Pmax * y
for i in I:
    for t in T:
        model.addConstr(p[i, t] >= Pmin[i] * y[i, t], name=f"GenMin_{i}_{t}")
        model.addConstr(p[i, t] <= Pmax[i] * y[i, t], name=f"GenMax_{i}_{t}")

# Startup / Shutdown logic: u - d = y_t - y_{t-1}
for i in I:
    for t in T:
        if t == 1:
            # y_{i,0} = y0[i]
            model.addConstr(u[i, t] - d[i, t] == y[i, t] - y0[i], name=f"StartupShutdown_{i}_{t}")
        else:
            model.addConstr(u[i, t] - d[i, t] == y[i, t] - y[i, t-1], name=f"StartupShutdown_{i}_{t}")

# Ramp/change linearization:
# Delta+ >= p_t - p_{t-1}
# Delta- >= p_{t-1} - p_t
for i in I:
    for t in T:
        if t == 1:
            prev_p = p0[i]
        else:
            prev_p = p[i, t-1]
        model.addConstr(Delta_pos[i, t] >= p[i, t] - prev_p, name=f"DeltaPos_{i}_{t}")
        model.addConstr(Delta_neg[i, t] >= prev_p - p[i, t], name=f"DeltaNeg_{i}_{t}")

# Demand satisfaction: sum_i p[i,t] >= D_t
for t in T:
    model.addConstr(gp.quicksum(p[i, t] for i in I) >= D[t], name=f"Demand_{t}")

# Combined-cycle persistence: y[i,t] <= y[i,t+1] for t=1..3
for i in combined_cycle_units:
    for t in range(1, number_of_time_periods):  # 1..3
        model.addConstr(y[i, t] <= y[i, t+1], name=f"CC_persistence_{i}_{t}")

# Minimum number of running units: sum_i y[i,t] >= min_units_running
for t in T:
    model.addConstr(gp.quicksum(y[i, t] for i in I) >= min_units_running,
                    name=f"MinUnits_{t}")

# ==========================
# 5. Objective Function
# ==========================

# Minimize total cost:
# sum_{t,i} [ fixed_cost_i * y[i,t]
#           + startup_cost_i * u[i,t]
#           + shutdown_cost_i * d[i,t]
#           + change_cost_i * (Delta_pos[i,t] + Delta_neg[i,t]) ]
obj_expr = gp.quicksum(
    fix_cost[i] * y[i, t]
    + start_cost[i] * u[i, t]
    + shut_cost[i] * d[i, t]
    + chg_cost[i] * (Delta_pos[i, t] + Delta_neg[i, t])
    for i in I for t in T
)

model.setObjective(obj_expr, GRB.MINIMIZE)

# ==========================
# 6. Optimize
# ==========================

model.Params.OutputFlag = 1  # set to 0 to silence Gurobi output
model.optimize()

# ==========================
# 7. Print Results
# ==========================

if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT or model.Status == GRB.SUBOPTIMAL:
    total_cost = model.ObjVal
    print(f"Optimal (or best found) total cost: {total_cost:.4f} USD\n")

    # Print basic schedule
    for t in T:
        print(f"Period {t}: Demand={D[t]}")
        running_units = [plant_names[i-1] for i in I if y[i, t].X > 0.5]
        total_generation = sum(p[i, t].X for i in I)
        print(f"  Total generation: {total_generation:.2f} MWh")
        print(f"  Running units ({len(running_units)}): {running_units}")
        print("  Outputs (MWh):")
        for i in I:
            if y[i, t].X > 0.5:
                print(f"    {plant_names[i-1]}: {p[i, t].X:.2f}")
        print()

    # Required final answer print (minimum total cost)
    print(f"FinalAnswer=【{total_cost:.4f}】")
else:
    print("Model did not solve to optimality or acceptable status.")
    print(f"FinalAnswer=【None】")