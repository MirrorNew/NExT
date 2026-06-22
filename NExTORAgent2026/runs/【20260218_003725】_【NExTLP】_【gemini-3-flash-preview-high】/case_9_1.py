import gurobipy as gp
from gurobipy import GRB

# 2. Define all parameter matrices and data inputs.
number_of_power_plants = 17
number_of_time_periods = 4
period_demands = {1: 11832.52, 2: 14467.92, 3: 16661.1, 4: 15434.44}
min_units_running = 8

plant_params = {
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

# 3. Create Model
model = gp.Model("Orissa_Electric_Holding_Company")

# 4. Create Decision Variables
plants = plant_params.keys()
periods = [1, 2, 3, 4]

y = model.addVars(plants, periods, vtype=GRB.BINARY, name="y")
u = model.addVars(plants, periods, vtype=GRB.BINARY, name="u")
d = model.addVars(plants, periods, vtype=GRB.BINARY, name="d")
p = model.addVars(plants, periods, vtype=GRB.CONTINUOUS, lb=0, name="p")
dp = model.addVars(plants, periods, vtype=GRB.CONTINUOUS, lb=0, name="dp")
dm = model.addVars(plants, periods, vtype=GRB.CONTINUOUS, lb=0, name="dm")

# 5. Set up the objective function
obj = gp.quicksum(
    plant_params[i]['fixed_cost'] * y[i,t] +
    plant_params[i]['startup_cost'] * u[i,t] +
    plant_params[i]['shutdown_cost'] * d[i,t] +
    plant_params[i]['change_cost'] * (dp[i,t] + dm[i,t])
    for i in plants for t in periods
)
model.setObjective(obj, GRB.MINIMIZE)

# 6. Add all constraints
# Demand satisfaction
for t in periods:
    model.addConstr(gp.quicksum(p[i,t] for i in plants) >= period_demands[t])

# Minimum operating units
for t in periods:
    model.addConstr(gp.quicksum(y[i,t] for i in plants) >= min_units_running)

# Plant logic and indicators
for i in plants:
    for t in periods:
        # Use addGenConstrIndicator for bounds based on online status
        model.addGenConstrIndicator(y[i,t], 1, p[i,t] >= plant_params[i]['min_power'])
        model.addGenConstrIndicator(y[i,t], 1, p[i,t] <= plant_params[i]['max_power'])
        model.addGenConstrIndicator(y[i,t], 0, p[i,t] <= 0)
        
        # Startup and Shutdown indicator definition
        y_prev = y[i, t-1] if t > 1 else 0
        model.addConstr(u[i,t] - d[i,t] == y[i,t] - y_prev)
        
        # Power output change definition (to capture |p_t - p_t-1|)
        p_prev = p[i, t-1] if t > 1 else 0
        model.addConstr(dp[i,t] - dm[i,t] == p[i,t] - p_prev)

# Combined Cycle units persistence: y[i,t] <= y[i,t+1]
cc_units = [i for i in plants if "Combined Cycle" in i]
for i in cc_units:
    for t in range(1, number_of_time_periods):
        model.addConstr(y[i,t] <= y[i,t+1])

# 7. Solve the model and print results
model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"FinalAnswer=【{model.objVal}】")