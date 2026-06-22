import gurobipy as gp
from gurobipy import GRB

# ===============================
# 1. Parameters from Parameters List
# ===============================

number_of_generators = 3
number_of_substations = 4
dispatch_periods = 5
maintenance_min_shutdowns_unit3 = 1
max_substations_per_unit = 2
max_consecutive_supply = 3
min_units_started = 2
fixed_equipment_fee = 500

Table_1_GeneratorParameters = {
    'units': [1, 2, 3],
    'P_min': [20, 30, 0],
    'P_max': [50, 80, 70],
    'R': [40, 30, 70],
    'c': [50, 60, 100]
}

Table_2_SubstationDemands = {
    'substations': [1, 2, 3, 4],
    'time_periods': [1, 2, 3, 4, 5],
    # D[j_index][t_index], j_index=0..3, t_index=0..4
    'D': [
        [40, 30, 60, 35, 50],  # Substation 1
        [30, 30, 40, 25, 40],  # Substation 2
        [50, 40, 50, 40, 30],  # Substation 3
        [30, 20, 30, 30, 40]   # Substation 4
    ]
}

# Derived sets
I = range(1, number_of_generators + 1)      # Units 1..3
J = range(1, number_of_substations + 1)     # Substations 1..4
T = range(1, dispatch_periods + 1)          # Time periods 1..5

# Map generator parameters by unit index i
P_min = {i: Table_1_GeneratorParameters['P_min'][i-1] for i in I}
P_max = {i: Table_1_GeneratorParameters['P_max'][i-1] for i in I}
R     = {i: Table_1_GeneratorParameters['R'][i-1]     for i in I}
c     = {i: Table_1_GeneratorParameters['c'][i-1]     for i in I}

# Demand D[j,t]
D = {}
for j in J:
    for t in T:
        D[j, t] = Table_2_SubstationDemands['D'][j-1][t-1]

# ===============================
# 2. Create model
# ===============================

model = gp.Model("Power_Dispatch_Unit_Commitment")

# ===============================
# 3. Decision variables
# ===============================

# u[i,t] : on/off status of unit i at time t (binary)
u = model.addVars(I, T, vtype=GRB.BINARY, name="u")

# P[i,t] : total power output of unit i at time t (continuous)
P = model.addVars(I, T, lb=0.0, vtype=GRB.CONTINUOUS, name="P")

# x[i,j,t] : assignment indicator (unit i supplies substation j at time t) (binary)
x = model.addVars(I, J, T, vtype=GRB.BINARY, name="x")

# P_ij[i,j,t] : power delivered from unit i to substation j at time t (continuous)
P_ij = model.addVars(I, J, T, lb=0.0, vtype=GRB.CONTINUOUS, name="P_ij")

# ===============================
# 4. Objective function
# ===============================

# Minimize sum of generation cost plus fixed equipment fee
gen_cost = gp.quicksum(c[i] * P[i, t] for i in I for t in T)
model.setObjective(gen_cost + fixed_equipment_fee, GRB.MINIMIZE)

# ===============================
# 5. Constraints
# ===============================

# C1: Output lower bound: P_{i,t} >= P_i^{min} * u_{i,t}
for i in I:
    for t in T:
        model.addConstr(P[i, t] >= P_min[i] * u[i, t], name=f"C1_min_output_{i}_{t}")

# C2: Output upper bound: P_{i,t} <= P_i^{max} * u_{i,t}
for i in I:
    for t in T:
        model.addConstr(P[i, t] <= P_max[i] * u[i, t], name=f"C2_max_output_{i}_{t}")

# C3: Coupling u–P: P_{i,t} = sum_j P_{i,j,t}
for i in I:
    for t in T:
        model.addConstr(
            P[i, t] == gp.quicksum(P_ij[i, j, t] for j in J),
            name=f"C3_balance_unit_{i}_{t}"
        )

# C4: Demand satisfaction: sum_i P_{i,j,t} = D_{j,t}
for j in J:
    for t in T:
        model.addConstr(
            gp.quicksum(P_ij[i, j, t] for i in I) == D[j, t],
            name=f"C4_demand_{j}_{t}"
        )

# C5: Assignment linking: P_{i,j,t} <= D_{j,t} * x_{i,j,t}
for i in I:
    for j in J:
        for t in T:
            model.addConstr(
                P_ij[i, j, t] <= D[j, t] * x[i, j, t],
                name=f"C5_link_{i}_{j}_{t}"
            )

# C6: Max. two substations per unit: sum_j x_{i,j,t} <= max_substations_per_unit
for i in I:
    for t in T:
        model.addConstr(
            gp.quicksum(x[i, j, t] for j in J) <= max_substations_per_unit,
            name=f"C6_max_substations_{i}_{t}"
        )

# C7 & C8: Ramp-up and ramp-down limits
# Assume P[i,0] = 0 to match initial off state
for i in I:
    # t = 1: implicitly ramp from 0, but no explicit constraint is strictly required.
    for t in T:
        if t > 1:
            # C7: Ramp-up: P_{i,t} - P_{i,t-1} <= R_i
            model.addConstr(
                P[i, t] - P[i, t-1] <= R[i],
                name=f"C7_ramp_up_{i}_{t}"
            )
            # C8: Ramp-down: P_{i,t-1} - P_{i,t} <= R_i
            model.addConstr(
                P[i, t-1] - P[i, t] <= R[i],
                name=f"C8_ramp_down_{i}_{t}"
            )

# C9: No four consecutive assignments for same (i,j)
# For t=4,5: sum_{tau=t-3..t} x_{i,j,tau} <= max_consecutive_supply
for i in I:
    for j in J:
        for t in T:
            if t >= max_consecutive_supply + 1:  # here 3+1=4 => t=4,5
                model.addConstr(
                    gp.quicksum(x[i, j, tau] for tau in range(t - max_consecutive_supply, t + 1))
                    <= max_consecutive_supply,
                    name=f"C9_no_four_consecutive_{i}_{j}_{t}"
                )

# C10: Unit 3 maintenance: sum_t u_{3,t} <= 5 - maintenance_min_shutdowns_unit3
max_on_periods_unit3 = dispatch_periods - maintenance_min_shutdowns_unit3
model.addConstr(
    gp.quicksum(u[3, t] for t in T) <= max_on_periods_unit3,
    name="C10_maintenance_unit3"
)

# C11: Spare-capacity: sum_i u_{i,t} >= min_units_started
for t in T:
    model.addConstr(
        gp.quicksum(u[i, t] for i in I) >= min_units_started,
        name=f"C11_min_units_started_{t}"
    )

# C12: Initial state u_{i,0}=0 is conceptual; not an explicit variable here.
# We already assume P[i,0]=0 for ramping.

# Variable domains (C13) are already handled by variable types and bounds.

# ===============================
# 6. Solve the model
# ===============================

model.optimize()

# ===============================
# 7. Print results
# ===============================

if model.status == GRB.OPTIMAL:
    print(f"Optimal objective value (total cost including fixed fee): {model.objVal}")

    # Print unit on/off status
    print("\nUnit on/off schedule (u[i,t]):")
    for i in I:
        for t in T:
            print(f"u[{i},{t}] = {u[i, t].X:.0f}", end="  ")
        print()

    # Print unit outputs
    print("\nUnit outputs P[i,t] (MW):")
    for i in I:
        for t in T:
            print(f"P[{i},{t}] = {P[i, t].X:.2f}", end="  ")
        print()

    # Print assignment decisions
    print("\nAssignments x[i,j,t]:")
    for t in T:
        print(f"\nTime period {t}:")
        for i in I:
            for j in J:
                if x[i, j, t].X > 0.5:
                    print(f"  Unit {i} supplies Substation {j}")

    # Print power flows to substations
    print("\nPower delivered P_ij[i,j,t] (MW):")
    for t in T:
        print(f"\nTime period {t}:")
        for i in I:
            for j in J:
                val = P_ij[i, j, t].X
                if val > 1e-6:
                    print(f"  P_ij[{i},{j},{t}] = {val:.2f}")

    # Final answer is the total cost including fixed equipment fee
    final_answer = model.objVal
else:
    print(f"Optimization ended with status {model.status}")
    final_answer = None

# ===============================
# 8. Required final answer print
# ===============================

print(f"FinalAnswer=【{final_answer}】")