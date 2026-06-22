import gurobipy as gp

# 1. Define parameters
num_generators = 3
num_substations = 4
num_periods = 5
fixed_fee = 500

# Generator parameters (index 0 corresponds to unit 1)
P_min = [20, 30, 0]  # MW
P_max = [50, 80, 70]  # MW
R = [40, 30, 70]  # MW/hour ramp limits
c = [50, 60, 100]  # $/MWh

# Demand data D[j][t] where j is substation index (0-based), t is time period (0-based)
D = [
    [40, 30, 60, 35, 50],    # Substation 1 demands
    [30, 30, 40, 25, 40],    # Substation 2 demands
    [50, 40, 50, 40, 30],    # Substation 3 demands
    [30, 20, 30, 30, 40]     # Substation 4 demands
]

# 2. Create model
model = gp.Model("PowerDispatch")

# 3. Decision variables
u = {}  # u[i,t] - ON/OFF status
P = {}  # P[i,t] - Power output
x = {}  # x[i,j,t] - Assignment indicator
Pflow = {}  # Pflow[i,j,t] - Power flow

# Create variables
for i in range(num_generators):
    for t in range(num_periods):
        u[i,t] = model.addVar(vtype=gp.GRB.BINARY, name=f"u_{i+1}_{t+1}")
        P[i,t] = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"P_{i+1}_{t+1}")
        
        for j in range(num_substations):
            x[i,j,t] = model.addVar(vtype=gp.GRB.BINARY, name=f"x_{i+1}_{j+1}_{t+1}")
            Pflow[i,j,t] = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"Pflow_{i+1}_{j+1}_{t+1}")

# 4. Set objective function
obj = gp.quicksum(c[i] * P[i,t] for i in range(num_generators) for t in range(num_periods)) + fixed_fee
model.setObjective(obj, gp.GRB.MINIMIZE)

# 5. Add constraints

# C1: Output lower bound
for i in range(num_generators):
    for t in range(num_periods):
        model.addConstr(P[i,t] >= P_min[i] * u[i,t], name=f"C1_lower_{i+1}_{t+1}")

# C2: Output upper bound
for i in range(num_generators):
    for t in range(num_periods):
        model.addConstr(P[i,t] <= P_max[i] * u[i,t], name=f"C2_upper_{i+1}_{t+1}")

# C3: Coupling between total output and flows
for i in range(num_generators):
    for t in range(num_periods):
        model.addConstr(P[i,t] == gp.quicksum(Pflow[i,j,t] for j in range(num_substations)), 
                       name=f"C3_coupling_{i+1}_{t+1}")

# C4: Demand satisfaction
for j in range(num_substations):
    for t in range(num_periods):
        model.addConstr(gp.quicksum(Pflow[i,j,t] for i in range(num_generators)) == D[j][t], 
                       name=f"C4_demand_{j+1}_{t+1}")

# C5: Linking Pflow and x
for i in range(num_generators):
    for j in range(num_substations):
        for t in range(num_periods):
            model.addConstr(Pflow[i,j,t] <= D[j][t] * x[i,j,t], 
                           name=f"C5_link_{i+1}_{j+1}_{t+1}")

# C6: Max two substations per unit per period
for i in range(num_generators):
    for t in range(num_periods):
        model.addConstr(gp.quicksum(x[i,j,t] for j in range(num_substations)) <= 2, 
                       name=f"C6_maxsub_{i+1}_{t+1}")

# C7 & C8: Ramp-up and ramp-down limits (for t >= 2)
# Note: Initial state u_{i,0} = 0 and P_{i,0} = 0
for i in range(num_generators):
    for t in range(1, num_periods):  # t = 1,...,4 (0-based indexing)
        # Ramp-up: P[i,t] - P[i,t-1] <= R[i]
        model.addConstr(P[i,t] - P[i,t-1] <= R[i], 
                       name=f"C7_rampup_{i+1}_{t+1}")
        # Ramp-down: P[i,t-1] - P[i,t] <= R[i]
        model.addConstr(P[i,t-1] - P[i,t] <= R[i], 
                       name=f"C8_rampdown_{i+1}_{t+1}")

# C9: No four consecutive assignments to same substation
for i in range(num_generators):
    for j in range(num_substations):
        # For t = 4 (period 5 in 1-based, index 3 in 0-based)
        model.addConstr(x[i,j,0] + x[i,j,1] + x[i,j,2] + x[i,j,3] <= 3, 
                       name=f"C9_cons1_{i+1}_{j+1}_4")
        # For t = 5 (period 5 in 1-based, index 4 in 0-based)
        model.addConstr(x[i,j,1] + x[i,j,2] + x[i,j,3] + x[i,j,4] <= 3, 
                       name=f"C9_cons2_{i+1}_{j+1}_5")

# C10: Unit 3 maintenance requirement (unit 3 index is 2)
model.addConstr(gp.quicksum(u[2,t] for t in range(num_periods)) <= 4, 
               name="C10_maintenance")

# C11: Spare-capacity requirement (at least 2 units ON each period)
for t in range(num_periods):
    model.addConstr(gp.quicksum(u[i,t] for i in range(num_generators)) >= 2, 
                   name=f"C11_spare_{t+1}")

# No need to explicitly add constraint C12 (initial state u_{i,0}=0) 
# since we're only defining variables for t=1..5

# 6. Solve the model
model.optimize()

# 7. Print results
print("Optimization status:", model.status)
if model.status == gp.GRB.OPTIMAL:
    print(f"Total cost (including fixed fee): ${model.ObjVal:.2f}")
    
    # Print unit status and outputs
    print("\nUnit status (1=ON, 0=OFF) and outputs (MW):")
    for i in range(num_generators):
        print(f"Unit {i+1}: ", end="")
        for t in range(num_periods):
            print(f"Period {t+1}: u={u[i,t].X:.0f}, P={P[i,t].X:.2f}MW | ", end="")
        print()
    
    # Print power flows
    print("\nPower flows (MW) from units to substations:")
    for t in range(num_periods):
        print(f"\nPeriod {t+1}:")
        for i in range(num_generators):
            for j in range(num_substations):
                if Pflow[i,j,t].X > 0.001:
                    print(f"  Unit {i+1} -> Substation {j+1}: {Pflow[i,j,t].X:.2f}MW")
    
    # Print total generation cost (excluding fixed fee)
    gen_cost = sum(c[i] * P[i,t].X for i in range(num_generators) for t in range(num_periods))
    print(f"\nGeneration cost (excluding fixed fee): ${gen_cost:.2f}")
    print(f"Fixed equipment fee: ${fixed_fee}")
    print(f"Total cost: ${gen_cost + fixed_fee:.2f}")
    
    # Verify unit 3 maintenance
    unit3_on_periods = sum(u[2,t].X for t in range(num_periods))
    print(f"\nUnit 3 is ON in {unit3_on_periods} periods (must be ≤ 4 for maintenance): {unit3_on_periods <= 4}")
    
    # Verify spare capacity
    for t in range(num_periods):
        units_on = sum(u[i,t].X for i in range(num_generators))
        print(f"Period {t+1}: {units_on:.0f} units ON (must be ≥ 2): {units_on >= 2}")

print(f"FinalAnswer=【{model.ObjVal:.2f}】")