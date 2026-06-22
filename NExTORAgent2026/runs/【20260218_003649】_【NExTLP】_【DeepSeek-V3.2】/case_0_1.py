import gurobipy as gp
from gurobipy import GRB

# Create the model
model = gp.Model("RefineryOptimization")

# Define parameters using the provided Parameters List
octane_naphtha = {'light': 90, 'medium': 80, 'heavy': 70}
octane_reformed_gasoline = 115
yield_reform_gas_light_naphtha = 0.6
yield_reform_gas_medium_naphtha = 0.52
yield_reform_gas_heavy_naphtha = 0.45
octane_cracking_gasoline = 105
yield_pyrolysis_oil_light = 0.68
yield_pyrolysis_gasoline_light = 0.28
yield_pyrolysis_oil_heavy = 0.75
yield_pyrolysis_gasoline_heavy = 0.2
yield_lubricating_from_residual = 0.5
min_octane_premium_engine_oil = 94
min_octane_ordinary_engine_oil = 84
max_pressure_kerosene = 1.0
pressure_light_oil = 1.0
pressure_heavy_oil = 0.6
pressure_pyrolysis_oil = 1.5
pressure_residual_oil = 0.05
ratio_fuel_oil = {'light_oil': 10, 'heavy_oil': 3, 'cracking_oil': 4, 'residual_oil': 1}
avail_crude_oil_1 = 20000
avail_crude_oil_2 = 30000
cap_distillation = 45000
cap_reforming = 10000
cap_cracking = 8000
lubricating_oil_min = 500
lubricating_oil_max = 1000
min_ratio_premium_to_ordinary = 0.4
profit_premium_engine_oil = 700
profit_ordinary_engine_oil = 600
profit_kerosene = 400
profit_fuel_oil = 350
profit_lubricating_oil = 150
Table_1_C_1 = {
    'Crude oil 1': {'Light naphtha': 0.1, 'Medium naphtha': 0.2, 'Heavy naphtha': 0.2, 'Light oil': 0.12, 'Heavy oil': 0.2, 'Residue': 0.13},
    'Crude oil 2': {'Light naphtha': 0.15, 'Medium naphtha': 0.25, 'Heavy naphtha': 0.18, 'Light oil': 0.08, 'Heavy oil': 0.19, 'Residue': 0.12}
}

# Decision variables
D1 = model.addVar(lb=0, ub=avail_crude_oil_1, name="D1")
D2 = model.addVar(lb=0, ub=avail_crude_oil_2, name="D2")

# Distillation products
N_L = model.addVar(lb=0, name="N_L")
N_M = model.addVar(lb=0, name="N_M")
N_H = model.addVar(lb=0, name="N_H")
LO = model.addVar(lb=0, name="LO")
HO = model.addVar(lb=0, name="HO")
R = model.addVar(lb=0, name="R")

# Naphtha allocation
N_L_to_mix = model.addVar(lb=0, name="N_L_to_mix")
N_L_to_reform = model.addVar(lb=0, name="N_L_to_reform")
N_M_to_mix = model.addVar(lb=0, name="N_M_to_mix")
N_M_to_reform = model.addVar(lb=0, name="N_M_to_reform")
N_H_to_mix = model.addVar(lb=0, name="N_H_to_mix")
N_H_to_reform = model.addVar(lb=0, name="N_H_to_reform")

# Reformed gasoline
RG_L = model.addVar(lb=0, name="RG_L")
RG_M = model.addVar(lb=0, name="RG_M")
RG_H = model.addVar(lb=0, name="RG_H")

# Light oil and heavy oil allocation
LO_to_mix = model.addVar(lb=0, name="LO_to_mix")
LO_to_crack = model.addVar(lb=0, name="LO_to_crack")
HO_to_mix = model.addVar(lb=0, name="HO_to_mix")
HO_to_crack = model.addVar(lb=0, name="HO_to_crack")

# Pyrolysis products
CO_LO = model.addVar(lb=0, name="CO_LO")
CG_LO = model.addVar(lb=0, name="CG_LO")
CO_HO = model.addVar(lb=0, name="CO_HO")
CG_HO = model.addVar(lb=0, name="CG_HO")
CO = model.addVar(lb=0, name="CO")
CG = model.addVar(lb=0, name="CG")

# Residue allocation
R_to_mix = model.addVar(lb=0, name="R_to_mix")
R_to_lub = model.addVar(lb=0, name="R_to_lub")
Lub = model.addVar(lb=0, name="Lub")

# Engine oil blending components - high-end
N_L_H = model.addVar(lb=0, name="N_L_H")
N_M_H = model.addVar(lb=0, name="N_M_H")
N_H_H = model.addVar(lb=0, name="N_H_H")
RG_L_H = model.addVar(lb=0, name="RG_L_H")
RG_M_H = model.addVar(lb=0, name="RG_M_H")
RG_H_H = model.addVar(lb=0, name="RG_H_H")
CG_H = model.addVar(lb=0, name="CG_H")

# Engine oil blending components - ordinary
N_L_O = model.addVar(lb=0, name="N_L_O")
N_M_O = model.addVar(lb=0, name="N_M_O")
N_H_O = model.addVar(lb=0, name="N_H_O")
RG_L_O = model.addVar(lb=0, name="RG_L_O")
RG_M_O = model.addVar(lb=0, name="RG_M_O")
RG_H_O = model.addVar(lb=0, name="RG_H_O")
CG_O = model.addVar(lb=0, name="CG_O")

# Final products
HE = model.addVar(lb=0, name="HE")
OE = model.addVar(lb=0, name="OE")
K = model.addVar(lb=0, name="K")
FO = model.addVar(lb=0, name="FO")

# Binary variable for kerosene/fuel oil split
y_K = model.addVar(vtype=GRB.BINARY, name="y_K")

# Set objective
model.setObjective(
    profit_premium_engine_oil * HE +
    profit_ordinary_engine_oil * OE +
    profit_kerosene * K +
    profit_fuel_oil * FO +
    profit_lubricating_oil * Lub,
    GRB.MAXIMIZE
)

# Distillation constraints
model.addConstr(N_L == Table_1_C_1['Crude oil 1']['Light naphtha'] * D1 + Table_1_C_1['Crude oil 2']['Light naphtha'] * D2, name="C1")
model.addConstr(N_M == Table_1_C_1['Crude oil 1']['Medium naphtha'] * D1 + Table_1_C_1['Crude oil 2']['Medium naphtha'] * D2, name="C2")
model.addConstr(N_H == Table_1_C_1['Crude oil 1']['Heavy naphtha'] * D1 + Table_1_C_1['Crude oil 2']['Heavy naphtha'] * D2, name="C3")
model.addConstr(LO == Table_1_C_1['Crude oil 1']['Light oil'] * D1 + Table_1_C_1['Crude oil 2']['Light oil'] * D2, name="C4")
model.addConstr(HO == Table_1_C_1['Crude oil 1']['Heavy oil'] * D1 + Table_1_C_1['Crude oil 2']['Heavy oil'] * D2, name="C5")
model.addConstr(R == Table_1_C_1['Crude oil 1']['Residue'] * D1 + Table_1_C_1['Crude oil 2']['Residue'] * D2, name="C6")

# Naphtha allocation constraints
model.addConstr(N_L_to_mix + N_L_to_reform == N_L, name="C7")
model.addConstr(N_M_to_mix + N_M_to_reform == N_M, name="C8")
model.addConstr(N_H_to_mix + N_H_to_reform == N_H, name="C9")

# Reformed gasoline constraints
model.addConstr(RG_L == yield_reform_gas_light_naphtha * N_L_to_reform, name="C10")
model.addConstr(RG_M == yield_reform_gas_medium_naphtha * N_M_to_reform, name="C11")
model.addConstr(RG_H == yield_reform_gas_heavy_naphtha * N_H_to_reform, name="C12")

# Light oil and heavy oil allocation constraints
model.addConstr(LO_to_mix + LO_to_crack == LO, name="C13")
model.addConstr(HO_to_mix + HO_to_crack == HO, name="C14")

# Pyrolysis constraints
model.addConstr(CO_LO == yield_pyrolysis_oil_light * LO_to_crack, name="C15")
model.addConstr(CG_LO == yield_pyrolysis_gasoline_light * LO_to_crack, name="C16")
model.addConstr(CO_HO == yield_pyrolysis_oil_heavy * HO_to_crack, name="C17")
model.addConstr(CG_HO == yield_pyrolysis_gasoline_heavy * HO_to_crack, name="C18")
model.addConstr(CO == CO_LO + CO_HO, name="C19")
model.addConstr(CG == CG_LO + CG_HO, name="C20")

# Residue allocation constraints
model.addConstr(R_to_mix + R_to_lub == R, name="C21")
model.addConstr(Lub == yield_lubricating_from_residual * R_to_lub, name="C22")

# Engine oil blending component allocation constraints
model.addConstr(N_L_H + N_L_O == N_L_to_mix, name="C23")
model.addConstr(N_M_H + N_M_O == N_M_to_mix, name="C24")
model.addConstr(N_H_H + N_H_O == N_H_to_mix, name="C25")
model.addConstr(RG_L_H + RG_L_O == RG_L, name="C26")
model.addConstr(RG_M_H + RG_M_O == RG_M, name="C27")
model.addConstr(RG_H_H + RG_H_O == RG_H, name="C28")
model.addConstr(CG_H + CG_O == CG, name="C29")

# Engine oil production constraints
model.addConstr(HE == N_L_H + N_M_H + N_H_H + RG_L_H + RG_M_H + RG_H_H + CG_H, name="C30")
model.addConstr(OE == N_L_O + N_M_O + N_H_O + RG_L_O + RG_M_O + RG_H_O + CG_O, name="C31")

# Octane constraints
model.addConstr(
    octane_naphtha['light'] * N_L_H + octane_naphtha['medium'] * N_M_H + octane_naphtha['heavy'] * N_H_H +
    octane_reformed_gasoline * (RG_L_H + RG_M_H + RG_H_H) +
    octane_cracking_gasoline * CG_H >= min_octane_premium_engine_oil * HE,
    name="C32"
)
model.addConstr(
    octane_naphtha['light'] * N_L_O + octane_naphtha['medium'] * N_M_O + octane_naphtha['heavy'] * N_H_O +
    octane_reformed_gasoline * (RG_L_O + RG_M_O + RG_H_O) +
    octane_cracking_gasoline * CG_O >= min_octane_ordinary_engine_oil * OE,
    name="C33"
)

# Kerosene/Fuel oil mixture
total_fuel_mix = LO_to_mix + HO_to_mix + CO + R_to_mix
model.addConstr(K + FO == total_fuel_mix, name="C34")

# Gas pressure constraint for kerosene (using indicator constraints)
# If K > 0 (y_K = 1), then pressure constraint must hold
M = 1e6  # Large number for big-M
model.addConstr(K <= M * y_K, name="C35_K_upper")
model.addConstr(FO <= M * (1 - y_K), name="C35_FO_upper")

# Pressure constraint: weighted average <= max_pressure_kerosene
pressure_expr = (pressure_light_oil * LO_to_mix + pressure_heavy_oil * HO_to_mix + 
                 pressure_pyrolysis_oil * CO + pressure_residual_oil * R_to_mix)

model.addGenConstrIndicator(y_K, 1, pressure_expr <= max_pressure_kerosene * K, name="C36_pressure")

# Fuel oil ratio constraints
model.addConstr(ratio_fuel_oil['light_oil'] * HO_to_mix == ratio_fuel_oil['heavy_oil'] * LO_to_mix, name="C37")
model.addConstr(ratio_fuel_oil['cracking_oil'] * HO_to_mix == ratio_fuel_oil['heavy_oil'] * CO, name="C38")
model.addConstr(ratio_fuel_oil['light_oil'] * R_to_mix == ratio_fuel_oil['residual_oil'] * LO_to_mix, name="C39")

# Capacity constraints
model.addConstr(D1 + D2 <= cap_distillation, name="C40")
model.addConstr(N_L_to_reform + N_M_to_reform + N_H_to_reform <= cap_reforming, name="C41")
model.addConstr(LO_to_crack + HO_to_crack <= cap_cracking, name="C42")

# Lubricating oil bounds
model.addConstr(Lub >= lubricating_oil_min, name="C43_lower")
model.addConstr(Lub <= lubricating_oil_max, name="C43_upper")

# High-end to ordinary engine oil ratio
model.addConstr(HE >= min_ratio_premium_to_ordinary * OE, name="C44")

# Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    total_profit = model.objVal
    print(f"Optimal total profit: {total_profit:.2f} yuan")
    print(f"FinalAnswer=【{total_profit:.2f}】")
else:
    print(f"Model status: {model.status}")
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible")
    print(f"FinalAnswer=【0】")