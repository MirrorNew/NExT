import gurobipy as gp
from gurobipy import GRB

# Solve the given MILP power dispatch problem with Gurobi
# and print the final total cost (including the fixed fee)
# in the required format: FinalAnswer=【value】

# -----------------------------
# 1. Define all parameter matrices and data inputs
# -----------------------------
num_units = 3
units = ['Unit 1', 'Unit 2', 'Unit 3']

P_min = {'Unit 1': 20, 'Unit 2': 30, 'Unit 3': 0}
P_max = {'Unit 1': 50, 'Unit 2': 80, 'Unit 3': 70}
R = {'Unit 1': 40, 'Unit 2': 30, 'Unit 3': 70}
c = {'Unit 1': 50, 'Unit 2': 60, 'Unit 3': 100}

num_periods = 5
periods = [1, 2, 3, 4, 5]

num_substations = 4
substations = ['Substation 1', 'Substation 2', 'Substation 3', 'Substation 4']

# Demand D[s][t] with t as string keys '1'..'5' exactly as given
D = {
    'Substation 1': {'1': 40, '2': 30, '3': 60, '4': 35, '5': 50},
    'Substation 2': {'1': 30, '2': 30, '3': 40, '4': 25, '5': 40},
    'Substation 3': {'1': 50, '2': 40, '3': 50, '4': 40, '5': 30},
    'Substation 4': {'1': 30, '2': 20, '3': 30, '4': 30, '5': 40},
}

fixed_equipment_fee = 500

# Big-M for linking x and z; must be >= maximum possible x
M = max(P_max.values())  # 80

# -----------------------------
# 2. Create model
# -----------------------------
model = gp.Model("Power_Dispatch_Optimization")

# -----------------------------
# 3. Create decision variables
# -----------------------------
# u[i,t]: ON/OFF status of unit i in period t (binary)
u = model.addVars(units, periods, vtype=GRB.BINARY, name="u")

# P[i,t]: total power output of unit i in period t (continuous, >= 0)
P = model.addVars(units, periods, lb=0.0, vtype=GRB.CONTINUOUS, name="P")

# x[i,s,t]: power from unit i to substation s at time t (continuous, >= 0)
x = model.addVars(units, substations, periods, lb=0.0, vtype=GRB.CONTINUOUS, name="x")

# z[i,s,t]: supply indicator: 1 if unit i supplies substation s in period t, 0 otherwise (binary)
z = model.addVars(units, substations, periods, vtype=GRB.BINARY, name="z")

# Δ^+_{i,t}, Δ^-_{i,t}: ramp-up and ramp-down amounts (continuous, >= 0)
Delta_plus = model.addVars(units, periods, lb=0.0, vtype=GRB.CONTINUOUS, name="Delta_plus")
Delta_minus = model.addVars(units, periods, lb=0.0, vtype=GRB.CONTINUOUS, name="Delta_minus")

# -----------------------------
# 4. Set up the objective function
# -----------------------------
gen_cost = gp.quicksum(c[i] * P[i, t] for i in units for t in periods)
model.setObjective(gen_cost + fixed_equipment_fee, GRB.MINIMIZE)

# -----------------------------
# 5. Add all constraints
# -----------------------------

# 5.1 Min/Max output bounds & coupling: P_i^{min} u_{i,t} ≤ P_{i,t} ≤ P_i^{max} u_{i,t}
for i in units:
    for t in periods:
        model.addConstr(P_min[i] * u[i, t] <= P[i, t],
                        name=f"min_output_{i}_{t}")
        model.addConstr(P[i, t] <= P_max[i] * u[i, t],
                        name=f"max_output_{i}_{t}")

# 5.2 Unit 3 maintenance shutdown at least once: ∑_{t=1}^5 u_{3,t} ≤ 4
model.addConstr(
    gp.quicksum(u['Unit 3', t] for t in periods) <= 4,
    name="maintenance_unit3"
)

# 5.3 Substation demand satisfaction: ∑_i x_{i,s,t} = D_{s,t}
for s in substations:
    for t in periods:
        model.addConstr(
            gp.quicksum(x[i, s, t] for i in units) == D[s][str(t)],
            name=f"demand_{s}_{t}"
        )

# 5.4 Unit output equals total dispatched to substations: ∑_s x_{i,s,t} = P_{i,t}
for i in units:
    for t in periods:
        model.addConstr(
            gp.quicksum(x[i, s, t] for s in substations) == P[i, t],
            name=f"balance_{i}_{t}"
        )

# 5.5 Ramp balance: P_{i,t} - P_{i,t-1} = Δ^+_{i,t} - Δ^-_{i,t}, for t = 2..5
for i in units:
    for t in periods:
        if t == 1:
            continue
        prev_t = t - 1
        model.addConstr(
            P[i, t] - P[i, prev_t] == Delta_plus[i, t] - Delta_minus[i, t],
            name=f"ramp_balance_{i}_{t}"
        )

# 5.6 Ramp-up/down limit:
# Δ^+_{i,t} ≥ 0, Δ^-_{i,t} ≥ 0 already enforced by variable lb
# and Δ^+_{i,t} + Δ^-_{i,t} ≤ R_i, for t = 2..5 (as in validated model)
for i in units:
    for t in periods:
        if t == 1:
            continue
        model.addConstr(
            Delta_plus[i, t] + Delta_minus[i, t] <= R[i],
            name=f"ramp_limit_{i}_{t}"
        )

# 5.7 Initial OFF status and zero initial output:
# In the validated model, u_{i,0}=0 and P_{i,0}=0 are given as initial conditions.
# We do not create variables for t=0; the ramp equations are enforced from t=2,
# so t=1 is free relative to the initial OFF state, consistent with the provided model.

# 5.8 At least two units ON every period: ∑_i u_{i,t} ≥ 2
for t in periods:
    model.addConstr(
        gp.quicksum(u[i, t] for i in units) >= 2,
        name=f"min_on_units_{t}"
    )

# 5.9 At most two substations per unit per period: ∑_s z_{i,s,t} ≤ 2
for i in units:
    for t in periods:
        model.addConstr(
            gp.quicksum(z[i, s, t] for s in substations) <= 2,
            name=f"max_substations_{i}_{t}"
        )

# 5.10 Dispatch only if supply indicator is active (big-M link): x_{i,s,t} ≤ M z_{i,s,t}
for i in units:
    for s in substations:
        for t in periods:
            model.addConstr(
                x[i, s, t] <= M * z[i, s, t],
                name=f"link_x_z_{i}_{s}_{t}"
            )

# 5.11 No supply if unit is OFF (linking x and u): x_{i,s,t} ≤ P_i^{max} u_{i,t}
for i in units:
    for s in substations:
        for t in periods:
            model.addConstr(
                x[i, s, t] <= P_max[i] * u[i, t],
                name=f"link_x_u_{i}_{s}_{t}"
            )

# 5.12 Three-consecutive-period supply prohibition:
# For all i, s and t=1,2: z_{i,s,t} + z_{i,s,t+1} + z_{i,s,t+2} + z_{i,s,t+3} ≤ 3
for i in units:
    for s in substations:
        for t in [1, 2]:
            model.addConstr(
                z[i, s, t] + z[i, s, t + 1] + z[i, s, t + 2] + z[i, s, t + 3] <= 3,
                name=f"no_four_consecutive_{i}_{s}_{t}"
            )

# -----------------------------
# 6. Solve the model
# -----------------------------
model.optimize()

# -----------------------------
# 7. Print results and FinalAnswer
# -----------------------------
if model.status == GRB.OPTIMAL:
    # Print some basic info
    print(f"Optimal objective value (total cost including fixed fee): {model.objVal:.2f}")
    for t in periods:
        print(f"\nTime period {t}:")
        for i in units:
            print(f"  {i}: u={int(round(u[i, t].X))}, P={P[i, t].X:.2f}")
        print("  Substation supplies:")
        for s in substations:
            line_parts = []
            for i in units:
                val = x[i, s, t].X
                if val > 1e-6:
                    line_parts.append(f"{i}: {val:.2f}")
            print(f"    {s} <- " + (", ".join(line_parts) if line_parts else "none"))

    # Final answer is the total cost (objective value)
    final_answer = model.objVal
else:
    # If not optimal, still define a final answer (could be None or a status code)
    final_answer = None
    print(f"Optimization did not reach optimality. Status: {model.status}")

# Required final output line
print(f"FinalAnswer=【{final_answer}】")