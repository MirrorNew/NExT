import gurobipy as gp
from gurobipy import GRB

# ====== 1. Parameters ======
# From Parameters List
num_plants = 17
num_periods = 4
period_length = 6
initial_output = 0
demands = [11832.52, 14467.92, 16661.10, 15434.44]  # periods 1-4
transmission_loss = 0
min_units_running = 8
initial_state = 0

# Power plant data from Table_2_Power_plant_parameters
plant_data = {
    'Biomass 1': {'min': 78.75, 'max': 295.14, 'change': 3.94, 'fixed': 265.524, 'startup': 0, 'shutdown': 330},
    'Biomass 2': {'min': 100.0, 'max': 300.0, 'change': 4.1, 'fixed': 270.0, 'startup': 0, 'shutdown': 340},
    'Combined Cycle 1': {'min': 742.5, 'max': 2883.9, 'change': 2.72, 'fixed': 92.28, 'startup': 500, 'shutdown': 210},
    'Combined Cycle 2': {'min': 800.0, 'max': 3000.0, 'change': 2.85, 'fixed': 95.0, 'startup': 500, 'shutdown': 215},
    'Combined Cycle 3': {'min': 700.0, 'max': 2800.0, 'change': 2.65, 'fixed': 90.0, 'startup': 500, 'shutdown': 205},
    'Combined Cycle 4': {'min': 750.0, 'max': 2900.0, 'change': 2.8, 'fixed': 93.0, 'startup': 500, 'shutdown': 208},
    'Wind Energy 1': {'min': 246.0, 'max': 294.9, 'change': 0.0, 'fixed': 149.778, 'startup': 0, 'shutdown': 240},
    'Wind Energy 2': {'min': 200.0, 'max': 350.0, 'change': 0.0, 'fixed': 160.0, 'startup': 0, 'shutdown': 250},
    'Wind Energy 3': {'min': 220.0, 'max': 330.0, 'change': 0.0, 'fixed': 155.0, 'startup': 0, 'shutdown': 245},
    'Wind Energy 4': {'min': 250.0, 'max': 340.0, 'change': 0.0, 'fixed': 162.0, 'startup': 0, 'shutdown': 255},
    'Hydropower 1': {'min': 50.0, 'max': 500.0, 'change': 1.5, 'fixed': 60.0, 'startup': 0, 'shutdown': 100},
    'Hydropower 2': {'min': 60.0, 'max': 600.0, 'change': 1.8, 'fixed': 65.0, 'startup': 0, 'shutdown': 120},
    'Hydropower3': {'min': 55.0, 'max': 550.0, 'change': 1.6, 'fixed': 62.0, 'startup': 0, 'shutdown': 110},
    'Solar1': {'min': 20.0, 'max': 400.0, 'change': 0.0, 'fixed': 120.0, 'startup': 0, 'shutdown': 150},
    'Solar2': {'min': 20.0, 'max': 350.0, 'change': 0.0, 'fixed': 110.0, 'startup': 0, 'shutdown': 140},
    'Coal1': {'min': 500.0, 'max': 2500.0, 'change': 5.0, 'fixed': 320.0, 'startup': 500, 'shutdown': 400},
    'Coal2': {'min': 550.0, 'max': 2600.0, 'change': 4.8, 'fixed': 315.0, 'startup': 480, 'shutdown': 390}
}

# Extract plant names in order
plant_names = list(plant_data.keys())

# Create parameter arrays
Pmin = [plant_data[name]['min'] for name in plant_names]
Pmax = [plant_data[name]['max'] for name in plant_names]
FixedCost = [plant_data[name]['fixed'] for name in plant_names]
StartupCost = [plant_data[name]['startup'] for name in plant_names]
ShutdownCost = [plant_data[name]['shutdown'] for name in plant_names]
ChangeCost = [plant_data[name]['change'] for name in plant_names]

# Combined Cycle indices (0-based: indices 2,3,4,5)
CC_indices = [2, 3, 4, 5]  # Corresponding to Combined Cycle 1-4

# ====== 2. Create Model ======
model = gp.Model("Power_Plant_Dispatch")

# ====== 3. Decision Variables ======
# Binary variables
y = {}  # online status
u = {}  # startup indicator
d = {}  # shutdown indicator

# Continuous variables
p = {}  # power output
delta_plus = {}  # positive change
delta_minus = {}  # negative change

for i in range(num_plants):
    for t in range(num_periods):
        y[i, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{t}")
        u[i, t] = model.addVar(vtype=GRB.BINARY, name=f"u_{i}_{t}")
        d[i, t] = model.addVar(vtype=GRB.BINARY, name=f"d_{i}_{t}")
        p[i, t] = model.addVar(lb=0.0, ub=Pmax[i], name=f"p_{i}_{t}")
        delta_plus[i, t] = model.addVar(lb=0.0, name=f"delta_plus_{i}_{t}")
        delta_minus[i, t] = model.addVar(lb=0.0, name=f"delta_minus_{i}_{t}")

# ====== 4. Objective Function ======
obj_expr = gp.QuadExpr()
for t in range(num_periods):
    for i in range(num_plants):
        obj_expr += FixedCost[i] * y[i, t]
        obj_expr += StartupCost[i] * u[i, t]
        obj_expr += ShutdownCost[i] * d[i, t]
        obj_expr += ChangeCost[i] * (delta_plus[i, t] + delta_minus[i, t])

model.setObjective(obj_expr, GRB.MINIMIZE)

# ====== 5. Constraints ======
# 5.1 Generation bounds
for i in range(num_plants):
    for t in range(num_periods):
        model.addConstr(p[i, t] >= Pmin[i] * y[i, t], name=f"min_power_{i}_{t}")
        model.addConstr(p[i, t] <= Pmax[i] * y[i, t], name=f"max_power_{i}_{t}")

# 5.2 Startup/Shutdown definition (with initial y_{i,0}=0)
for i in range(num_plants):
    for t in range(num_periods):
        if t == 0:
            # y[i,-1] = 0 (initial state)
            model.addConstr(u[i, t] - d[i, t] == y[i, t] - 0, name=f"start_shutdown_{i}_{t}")
        else:
            model.addConstr(u[i, t] - d[i, t] == y[i, t] - y[i, t-1], name=f"start_shutdown_{i}_{t}")

# 5.3 Ramp change linearization (with initial p_{i,0}=0)
for i in range(num_plants):
    for t in range(num_periods):
        if t == 0:
            model.addConstr(delta_plus[i, t] >= p[i, t] - 0, name=f"ramp_plus_{i}_{t}")
            model.addConstr(delta_minus[i, t] >= 0 - p[i, t], name=f"ramp_minus_{i}_{t}")
        else:
            model.addConstr(delta_plus[i, t] >= p[i, t] - p[i, t-1], name=f"ramp_plus_{i}_{t}")
            model.addConstr(delta_minus[i, t] >= p[i, t-1] - p[i, t], name=f"ramp_minus_{i}_{t}")

# 5.4 Demand satisfaction
for t in range(num_periods):
    demand_expr = gp.quicksum(p[i, t] for i in range(num_plants))
    model.addConstr(demand_expr >= demands[t], name=f"demand_{t}")

# 5.5 Combined-Cycle persistence
for idx in CC_indices:
    for t in range(num_periods - 1):  # t = 0,1,2 (since t+1 must exist)
        model.addConstr(y[idx, t] <= y[idx, t+1], name=f"CC_persistence_{idx}_{t}")

# 5.6 Minimum number of online units
for t in range(num_periods):
    online_expr = gp.quicksum(y[i, t] for i in range(num_plants))
    model.addConstr(online_expr >= min_units_running, name=f"min_units_{t}")

# ====== 6. Solve ======
model.optimize()

# ====== 7. Output Results ======
if model.status == GRB.OPTIMAL:
    print(f"Optimal total cost: ${model.ObjVal:.2f}")
    
    # Print detailed solution
    for t in range(num_periods):
        total_power = sum(p[i, t].X for i in range(num_plants))
        online_units = sum(y[i, t].X for i in range(num_plants))
        print(f"\nPeriod {t+1} (Demand: {demands[t]:.2f} MWh):")
        print(f"  Total generation: {total_power:.2f} MWh, Online units: {online_units}")
        
        # Print significant generation (>0)
        for i in range(num_plants):
            if p[i, t].X > 0.1:
                print(f"    {plant_names[i]}: {p[i, t].X:.2f} MWh (Status: {int(y[i, t].X)})")
    
    # Output the minimum cost as required
    print(f"\nFinalAnswer=【{model.ObjVal:.2f}】")
else:
    print("No optimal solution found.")
    print(f"FinalAnswer=【No feasible solution】")