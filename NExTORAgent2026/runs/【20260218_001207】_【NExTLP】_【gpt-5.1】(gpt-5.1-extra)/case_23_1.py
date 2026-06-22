import gurobipy as gp
from gurobipy import GRB

# ===============================
# 1. Import and Parameters
# ===============================

# Parameters List (values must be used as given)
Products = ['A', 'B', 'C', 'D']
Months = ['January', 'February', 'March', 'April', 'May', 'June']
Planning_horizon_months = 6

Table_1_Monthly_demand = {
    'months': ['January', 'February', 'March', 'April', 'May', 'June'],
    'products': ['A', 'B', 'C', 'D'],
    'demand': {
        'A': [20, 22, 25, 18, 15, 10],
        'B': [30, 28, 35, 32, 25, 20],
        'C': [15, 20, 30, 40, 20, 20],
        'D': [15, 20, 30, 20, 10, 10]
    },
    'total_market_demand': [80, 90, 120, 110, 70, 60]
}

Demand = {
    'A': [20, 22, 25, 18, 15, 10],
    'B': [30, 28, 35, 32, 25, 20],
    'C': [15, 20, 30, 40, 20, 20],
    'D': [15, 20, 30, 20, 10, 10]
}

Total_market_demand = [80, 90, 120, 110, 70, 60]

Table_2_Monthly_production_capacity = {
    'products': ['A', 'B', 'C', 'D'],
    'regular_capacity_limit': [30, 30, 20, 20],
    'overtime_capacity_limit': [6, 6, 4, 4],
    'regular_unit_price': [9.0, 11.0, 13.0, 10.0],
    'overtime_unit_price': [13.5, 16.5, 19.5, 15.0]
}

Regular_capacity_limit = {'A': 30, 'B': 30, 'C': 20, 'D': 20}
Overtime_capacity_limit = {'A': 6, 'B': 6, 'C': 4, 'D': 4}

Regular_unit_price = {'A': 9.0, 'B': 11.0, 'C': 13.0, 'D': 10.0}
Overtime_unit_price = {'A': 13.5, 'B': 16.5, 'C': 19.5, 'D': 15.0}

Free_B_from_A_ratio = 2
Extra_B_cost_premium = 1.0
Expansion_threshold_CD_total_production = 40
Expanded_regular_capacity_C = 36
Expanded_regular_capacity_D = 36
Unit_price_reduction_after_expansion = 1.0
Overtime_price_factor_before_minus = 1.5
Overtime_price_reduction_after_expansion = 1.0

Initial_inventory = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
Production_and_inventory_integer = 1
Production_and_inventory_nonnegative = 1

T = Planning_horizon_months

# Inventory holding cost parameters h_{i,t}.
# Not explicitly given, so we treat them as 0 to avoid introducing new values.
h = {(i, t): 0.0 for i in Products for t in range(T)}

# Big-M and epsilon for trigger constraints (used as numeric constants)
M_big = 10000.0
eps = 1e-4

# ===============================
# 2. Create Gurobi Model
# ===============================

model = gp.Model("Jinling_Production_Inventory_Planning")

# ===============================
# 3. Decision Variables
# ===============================

# Regular production R_{i,t}
R = model.addVars(Products, range(T),
                  vtype=GRB.INTEGER if Production_and_inventory_integer else GRB.CONTINUOUS,
                  lb=0 if Production_and_inventory_nonnegative else -GRB.INFINITY,
                  name="R")

# Overtime production O_{i,t}
O = model.addVars(Products, range(T),
                  vtype=GRB.INTEGER if Production_and_inventory_integer else GRB.CONTINUOUS,
                  lb=0 if Production_and_inventory_nonnegative else -GRB.INFINITY,
                  name="O")

# Inventory I_{i,t}
I = model.addVars(Products, range(T),
                  vtype=GRB.INTEGER if Production_and_inventory_integer else GRB.CONTINUOUS,
                  lb=0 if Production_and_inventory_nonnegative else -GRB.INFINITY,
                  name="I")

# Binary indicator y_{i,t}
y = model.addVars(Products, range(T), vtype=GRB.BINARY, name="y")

# Free B from A by-product F_t
F = model.addVars(range(T),
                  vtype=GRB.INTEGER if Production_and_inventory_integer else GRB.CONTINUOUS,
                  lb=0 if Production_and_inventory_nonnegative else -GRB.INFINITY,
                  name="F")

# Total production P_{A,t}, P_{B,t}
PA = model.addVars(range(T),
                   vtype=GRB.INTEGER if Production_and_inventory_integer else GRB.CONTINUOUS,
                   lb=0 if Production_and_inventory_nonnegative else -GRB.INFINITY,
                   name="PA")
PB = model.addVars(range(T),
                   vtype=GRB.INTEGER if Production_and_inventory_integer else GRB.CONTINUOUS,
                   lb=0 if Production_and_inventory_nonnegative else -GRB.INFINITY,
                   name="PB")

# Δ_t^+, Δ_t^-
Delta_plus = model.addVars(range(T),
                           vtype=GRB.INTEGER if Production_and_inventory_integer else GRB.CONTINUOUS,
                           lb=0 if Production_and_inventory_nonnegative else -GRB.INFINITY,
                           name="Delta_plus")
Delta_minus = model.addVars(range(T),
                            vtype=GRB.INTEGER if Production_and_inventory_integer else GRB.CONTINUOUS,
                            lb=0 if Production_and_inventory_nonnegative else -GRB.INFINITY,
                            name="Delta_minus")

# e_{B,t}
eB = model.addVars(range(T),
                   vtype=GRB.INTEGER if Production_and_inventory_integer else GRB.CONTINUOUS,
                   lb=0 if Production_and_inventory_nonnegative else -GRB.INFINITY,
                   name="eB")

# Expansion trigger u_t
u = model.addVars(range(T), vtype=GRB.BINARY, name="u")

# Expansion active z_t
z = model.addVars(range(T), vtype=GRB.BINARY, name="z")

# Effective unit costs
Cunit = model.addVars(Products, range(T), vtype=GRB.CONTINUOUS, lb=0.0, name="Cunit")
Cot = model.addVars(Products, range(T), vtype=GRB.CONTINUOUS, lb=0.0, name="Cot")

# ===============================
# 4. Constraints
# ===============================

# Demand satisfaction / Inventory balance
for i in Products:
    for t in range(T):
        prev_I = Initial_inventory[i] if t == 0 else I[i, t - 1]
        if i == 'B':
            model.addConstr(
                prev_I + R[i, t] + O[i, t] + F[t] - I[i, t] == Demand[i][t],
                name=f"InvBal_B_{t}"
            )
        else:
            model.addConstr(
                prev_I + R[i, t] + O[i, t] - I[i, t] == Demand[i][t],
                name=f"InvBal_{i}_{t}"
            )

# Regular capacity limits
for i in Products:
    for t in range(T):
        model.addConstr(R[i, t] <= Regular_capacity_limit[i],
                        name=f"RegCap_{i}_{t}")

# Overtime capacity limits
for i in Products:
    for t in range(T):
        model.addConstr(O[i, t] <= Overtime_capacity_limit[i],
                        name=f"OTCap_{i}_{t}")

# Overtime only when regular at maximum – link 1: R_{i,t} ≥ CapReg_i * y_{i,t}
for i in Products:
    for t in range(T):
        model.addConstr(R[i, t] >= Regular_capacity_limit[i] * y[i, t],
                        name=f"RegAtCap_{i}_{t}")

# Overtime only when regular at maximum – link 2: O_{i,t} ≤ CapOT_i * y_{i,t}
for i in Products:
    for t in range(T):
        model.addConstr(O[i, t] <= Overtime_capacity_limit[i] * y[i, t],
                        name=f"OTWhenRegCap_{i}_{t}")

# Free B from A by-product (quantity bound): F_t ≤ 2 * R_{A,t}
for t in range(T):
    model.addConstr(F[t] <= Free_B_from_A_ratio * R['A', t],
                    name=f"FreeB_from_Aqty_{t}")

# Free B only if A regular completed:
# Use indicator constraints instead of big-M: if y_A,t = 0 then F_t = 0
for t in range(T):
    model.addGenConstrIndicator(y['A', t], 0, F[t] == 0,
                                name=f"FreeB_only_if_AFull_{t}")

# Total production definitions
for t in range(T):
    model.addConstr(PA[t] == R['A', t] + O['A', t], name=f"PA_def_{t}")
    model.addConstr(PB[t] == R['B', t] + O['B', t] + F[t], name=f"PB_def_{t}")

# A and B production difference decomposition
for t in range(T):
    model.addConstr(PB[t] - PA[t] == Delta_plus[t] - Delta_minus[t],
                    name=f"DiffAB_{t}")

# Excess B over A definition
for t in range(T):
    model.addConstr(eB[t] == Delta_plus[t], name=f"ExcessB_{t}")

# C and D expansion trigger (upper and lower bound forms)
for t in range(T):
    sum_CD = R['C', t] + O['C', t] + R['D', t] + O['D', t]
    model.addConstr(sum_CD - Expansion_threshold_CD_total_production <= M_big * u[t],
                    name=f"CDTriggerUpper_{t}")
    model.addConstr(sum_CD >= Expansion_threshold_CD_total_production + eps
                    - M_big * (1 - u[t]),
                    name=f"CDTriggerLower_{t}")

# Expansion effect from next month: z_{t+1} ≥ u_t
for t in range(T - 1):
    model.addConstr(z[t + 1] >= u[t], name=f"ExpansionNext_{t}")

# Expanded regular capacity for C and D
for t in range(T):
    model.addConstr(
        R['C', t] <= Regular_capacity_limit['C'] * (1 - z[t]) +
        Expanded_regular_capacity_C * z[t],
        name=f"ExpCap_C_{t}"
    )
    model.addConstr(
        R['D', t] <= Regular_capacity_limit['D'] * (1 - z[t]) +
        Expanded_regular_capacity_D * z[t],
        name=f"ExpCap_D_{t}"
    )

# Effective unit cost for C and D (regular)
for t in range(T):
    model.addConstr(
        Cunit['C', t] == Regular_unit_price['C'] -
        Unit_price_reduction_after_expansion * z[t],
        name=f"Cunit_C_{t}"
    )
    model.addConstr(
        Cunit['D', t] == Regular_unit_price['D'] -
        Unit_price_reduction_after_expansion * z[t],
        name=f"Cunit_D_{t}"
    )

# Effective unit cost for C and D (overtime)
for t in range(T):
    model.addConstr(
        Cot['C', t] == Overtime_price_factor_before_minus * Cunit['C', t],
        name=f"Cot_C_{t}"
    )
    model.addConstr(
        Cot['D', t] == Overtime_price_factor_before_minus * Cunit['D', t],
        name=f"Cot_D_{t}"
    )

# Base unit costs for A and B (regular)
for t in range(T):
    model.addConstr(Cunit['A', t] == Regular_unit_price['A'],
                    name=f"Cunit_A_{t}")
    model.addConstr(Cunit['B', t] == Regular_unit_price['B'],
                    name=f"Cunit_B_{t}")

# Base unit costs for A and B (overtime)
for t in range(T):
    model.addConstr(Cot['A', t] == Overtime_unit_price['A'],
                    name=f"Cot_A_{t}")
    model.addConstr(Cot['B', t] == Overtime_unit_price['B'],
                    name=f"Cot_B_{t}")

# ===============================
# 5. Objective Function
# ===============================

prod_cost = gp.quicksum(
    Cunit[i, t] * R[i, t] + Cot[i, t] * O[i, t]
    for i in Products for t in range(T)
)

excess_B_cost = gp.quicksum(Extra_B_cost_premium * eB[t] for t in range(T))

inv_cost = gp.quicksum(h[i, t] * I[i, t] for i in Products for t in range(T))

model.setObjective(prod_cost + excess_B_cost + inv_cost, GRB.MINIMIZE)

# ===============================
# 6. Solve the model
# ===============================

model.optimize()

# ===============================
# 7. Print results and Final Answer
# ===============================

if model.Status == GRB.OPTIMAL:
    total_cost = model.ObjVal
    print("Optimal total cost:", total_cost)
else:
    total_cost = None
    print("No optimal solution found. Status:", model.Status)

print(f"FinalAnswer=【{total_cost}】")