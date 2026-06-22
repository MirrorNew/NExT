import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
# Parameters provided in the problem description
number_of_power_plants = 17
number_of_time_periods = 4
period_demands = [None, 11832.52, 14467.92, 16661.1, 15434.44] # Index 1 to 4
min_units_running = 8

# Power plant parameters (copied exactly from Parameters List)
power_plant_params = {
    'Biomass 1': {'min_power': 78.75, 'max_power': 295.14, 'change_cost': 3.94, 'fixed_cost': 265.524, 'startup_cost': 0, 'shutdown_cost': 330},
    'Biomass 2': {'min_power': 100.0, 'max_power': 300.0, 'change_cost': 4.1, 'fixed_cost': 270.0, 'startup_cost': 0, 'shutdown_cost': 340},
    'Combined Cycle 1': {'min_power': 742.5, 'max_power': 2883.9, 'change_cost': 2.72, 'fixed_cost': 92.28, 'startup_cost': 500, 'shutdown_cost': 210},
    'Combined Cycle 2': {'min_power': 800.0, 'max_power': 3000.0, 'change_cost': 2.85, 'fixed_cost': 95.0, 'startup_cost': 500, 'shutdown_cost': 215},
    'Combined Cycle 3': {'min_power': 700.0, 'max_power': 2800.0, 'change_cost': 2.65, 'fixed_cost': 90.0, 'startup_cost': 500, 'shutdown_cost': 205},
    'Combined Cycle 4': {'min_power': 750.0, 'max_power': 2900.0, 'change_cost': 2.8, 'fixed_cost': 93.0, 'startup_cost': 500, 'shutdown_cost': 208},
    'Wind Energy 1': {'min_power': 246.0, 'max_power': 294.9, 'change_cost': 0.0, 'fixed_cost': 149.778, 'startup_cost': 0, 'shutdown_cost': 240},
    'Wind Energy 2': {'min_power': 200.0, 'max_power': 350.0, 'change_cost': 0.0, 'fixed_cost': 160.0, 'startup_cost': 0, 'shutdown_cost': 250},
    'Wind Energy 3': {'min_power': 220.0, 'max_power': 330.0, 'change_cost': 0.0, 'fixed_cost': 155.0, 'startup_cost': 0, 'shutdown_cost': 245},
    'Wind Energy 4': {'min_power': 250.0, 'max_power': 340.0, 'change_cost': 0.0, 'fixed_cost': 162.0, 'startup_cost': 0, 'shutdown_cost': 255},
    'Hydropower 1': {'min_power': 50.0, 'max_power': 500.0, 'change_cost': 1.5, 'fixed_cost': 60.0, 'startup_cost': 0, 'shutdown_cost': 100},
    'Hydropower 2': {'min_power': 60.0, 'max_power': 600.0, 'change_cost': 1.8, 'fixed_cost': 65.0, 'startup_cost': 0, 'shutdown_cost': 120},
    'Hydropower3': {'min_power': 55.0, 'max_power': 550.0, 'change_cost': 1.6, 'fixed_cost': 62.0, 'startup_cost': 0, 'shutdown_cost': 110},
    'Solar1': {'min_power': 20.0, 'max_power': 400.0, 'change_cost': 0.0, 'fixed_cost': 120.0, 'startup_cost': 0, 'shutdown_cost': 150},
    'Solar2': {'min_power': 20.0, 'max_power': 350.0, 'change_cost': 0.0, 'fixed_cost': 110.0, 'startup_cost': 0, 'shutdown_cost': 140},
    'Coal1': {'min_power': 500.0, 'max_power': 2500.0, 'change_cost': 5.0, 'fixed_cost': 320.0, 'startup_cost': 500, 'shutdown_cost': 400},
    'Coal2': {'min_power': 550.0, 'max_power': 2600.0, 'change_cost': 4.8, 'fixed_cost': 315.0, 'startup_cost': 480, 'shutdown_cost': 390}
}

plant_names = list(power_plant_params.keys())
time_periods = [1, 2, 3, 4]

# Initialize Model
model = gp.Model("OrissaElectric")

# 2. Create decision variables.
# y: online status (Binary), p: power output (Continuous), u: startup (Binary), d: shutdown (Binary)
# dp: positive change in power, dm: negative change in power
y = {}
p = {}
u = {}
d = {}
dp = {}
dm = {}

for i in plant_names:
    for t in time_periods:
        y[i, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{t}")
        p[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"p_{i}_{t}")
        u[i, t] = model.addVar(vtype=GRB.BINARY, name=f"u_{i}_{t}")
        d[i, t] = model.addVar(vtype=GRB.BINARY, name=f"d_{i}_{t}")
        dp[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"dp_{i}_{t}")
        dm[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"dm_{i}_{t}")

# 3. Set up the objective function.
total_cost = 0
for i in plant_names:
    params = power_plant_params[i]
    for t in time_periods:
        # Cost = Fixed Cost * y + Startup Cost * u + Shutdown Cost * d + Change Cost * (delta+ + delta-)
        term_cost = (params['fixed_cost'] * y[i, t] +
                     params['startup_cost'] * u[i, t] +
                     params['shutdown_cost'] * d[i, t] +
                     params['change_cost'] * (dp[i, t] + dm[i, t]))
        total_cost += term_cost

model.setObjective(total_cost, GRB.MINIMIZE)

# 4. Add all constraints.

# Demand satisfaction constraints
for t in time_periods:
    model.addConstr(gp.quicksum(p[i, t] for i in plant_names) >= period_demands[t], name=f"Demand_{t}")

# Minimum operating units constraint
for t in time_periods:
    model.addConstr(gp.quicksum(y[i, t] for i in plant_names) >= min_units_running, name=f"MinUnits_{t}")

# Constraints for each plant and time period
for i in plant_names:
    params = power_plant_params[i]
    for t in time_periods:
        # Power output limits (semi-continuous behavior)
        model.addConstr(p[i, t] >= params['min_power'] * y[i, t], name=f"MinPower_{i}_{t}")
        model.addConstr(p[i, t] <= params['max_power'] * y[i, t], name=f"MaxPower_{i}_{t}")
        
        # Previous state definitions (Initial state at t=0 is OFF and 0 power)
        if t == 1:
            y_prev = 0
            p_prev = 0
        else:
            y_prev = y[i, t-1]
            p_prev = p[i, t-1]
            
        # Startup and Shutdown logic: y[t] - y[t-1] = u[t] - d[t]
        model.addConstr(y[i, t] - y_prev == u[i, t] - d[i, t], name=f"StartStopLogic_{i}_{t}")
        
        # Ramp/Change definition: p[t] - p[t-1] = dp[t] - dm[t]
        model.addConstr(p[i, t] - p_prev == dp[i, t] - dm[i, t], name=f"ChangeLogic_{i}_{t}")

# Combined Cycle persistence constraint
# If a CC unit is running in period t, it must continue in t+1. 
# Implies y[t] <= y[t+1] (cannot switch from 1 to 0).
cc_units = [name for name in plant_names if "Combined Cycle" in name]
for i in cc_units:
    for t in range(1, number_of_time_periods): # Loops t=1, 2, 3. (Compare t with t+1)
        model.addConstr(y[i, t] <= y[i, t+1], name=f"CC_Persistence_{i}_{t}")

# 5. Solve the model and print results.
model.optimize()

# Output the result
if model.status == GRB.OPTIMAL:
    final_answer = model.ObjVal
    print(f"FinalAnswer=【{final_answer}】")
else:
    print("FinalAnswer=【No Solution Found】")