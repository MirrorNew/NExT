import gurobipy as gp
from gurobipy import GRB

# ==========================
# 1. Define parameters
# ==========================

# Use only the provided Parameters List

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
    'Crude oil 1': {
        'Light naphtha': 0.1,
        'Medium naphtha': 0.2,
        'Heavy naphtha': 0.2,
        'Light oil': 0.12,
        'Heavy oil': 0.2,
        'Residue': 0.13
    },
    'Crude oil 2': {
        'Light naphtha': 0.15,
        'Medium naphtha': 0.25,
        'Heavy naphtha': 0.18,
        'Light oil': 0.08,
        'Heavy oil': 0.19,
        'Residue': 0.12
    }
}

# Distillation yields for convenience
yLD1 = Table_1_C_1['Crude oil 1']['Light naphtha']
yMD1 = Table_1_C_1['Crude oil 1']['Medium naphtha']
yHD1 = Table_1_C_1['Crude oil 1']['Heavy naphtha']
yLOD1 = Table_1_C_1['Crude oil 1']['Light oil']
yHOD1 = Table_1_C_1['Crude oil 1']['Heavy oil']
yRD1 = Table_1_C_1['Crude oil 1']['Residue']

yLD2 = Table_1_C_1['Crude oil 2']['Light naphtha']
yMD2 = Table_1_C_1['Crude oil 2']['Medium naphtha']
yHD2 = Table_1_C_1['Crude oil 2']['Heavy naphtha']
yLOD2 = Table_1_C_1['Crude oil 2']['Light oil']
yHOD2 = Table_1_C_1['Crude oil 2']['Heavy oil']
yRD2 = Table_1_C_1['Crude oil 2']['Residue']

# ==========================
# 2. Create model
# ==========================

model = gp.Model("Refinery_Production_Flow")

# ==========================
# 3. Decision variables
# ==========================

# Crude processed
D1 = model.addVar(lb=0.0, ub=avail_crude_oil_1, vtype=GRB.CONTINUOUS, name="D1")
D2 = model.addVar(lb=0.0, ub=avail_crude_oil_2, vtype=GRB.CONTINUOUS, name="D2")

# Distillation outputs
N_L = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="N_L")
N_M = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="N_M")
N_H = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="N_H")
LO = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="LO")
HO = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="HO")
R = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="R")

# Naphtha allocation
N_L_to_mix = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="N_L_to_mix")
N_L_to_reform = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="N_L_to_reform")
N_M_to_mix = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="N_M_to_mix")
N_M_to_reform = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="N_M_to_reform")
N_H_to_mix = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="N_H_to_mix")
N_H_to_reform = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="N_H_to_reform")

# Reforming outputs
RG_L = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="RG_L")
RG_M = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="RG_M")
RG_H = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="RG_H")

# Light/heavy oil allocation
LO_to_mix = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="LO_to_mix")
LO_to_crack = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="LO_to_crack")
HO_to_mix = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="HO_to_mix")
HO_to_crack = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="HO_to_crack")

# Cracking outputs
CO_LO = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="CO_LO")
CG_LO = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="CG_LO")
CO_HO = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="CO_HO")
CG_HO = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="CG_HO")
CO = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="CO")
CG = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="CG")

# Residue allocation and lub
R_to_mix = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="R_to_mix")
R_to_lub = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="R_to_lub")
Lub = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="Lub")

# Engine-oil blending allocation (premium / ordinary)
N_L_H = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="N_L_H")
N_M_H = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="N_M_H")
N_H_H = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="N_H_H")
RG_L_H = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="RG_L_H")
RG_M_H = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="RG_M_H")
RG_H_H = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="RG_H_H")
CG_H = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="CG_H")

N_L_O = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="N_L_O")
N_M_O = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="N_M_O")
N_H_O = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="N_H_O")
RG_L_O = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="RG_L_O")
RG_M_O = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="RG_M_O")
RG_H_O = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="RG_H_O")
CG_O = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="CG_O")

# Engine-oil outputs
HE = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="HE")
OE = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="OE")

# Kerosene and fuel oil outputs
K = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="K")
FO = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="FO")

# Additional variables for fuel-oil ratio (FO components and scaling factor)
lambda_FO = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="lambda_FO")
FO_LO = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="FO_LO")  # light oil in FO
FO_HO = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="FO_HO")  # heavy oil in FO
FO_CO = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="FO_CO")  # cracking oil in FO
FO_R = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="FO_R")    # residual oil in FO

# ==========================
# 4. Constraints
# ==========================

# C1–C6: Distillation yields
model.addConstr(N_L == yLD1 * D1 + yLD2 * D2, name="C1")
model.addConstr(N_M == yMD1 * D1 + yMD2 * D2, name="C2")
model.addConstr(N_H == yHD1 * D1 + yHD2 * D2, name="C3")
model.addConstr(LO == yLOD1 * D1 + yLOD2 * D2, name="C4")
model.addConstr(HO == yHOD1 * D1 + yHOD2 * D2, name="C5")
model.addConstr(R == yRD1 * D1 + yRD2 * D2, name="C6")

# C7–C9: Naphtha allocation
model.addConstr(N_L_to_mix + N_L_to_reform == N_L, name="C7")
model.addConstr(N_M_to_mix + N_M_to_reform == N_M, name="C8")
model.addConstr(N_H_to_mix + N_H_to_reform == N_H, name="C9")

# C10–C12: Reforming yields
model.addConstr(RG_L == yield_reform_gas_light_naphtha * N_L_to_reform, name="C10")
model.addConstr(RG_M == yield_reform_gas_medium_naphtha * N_M_to_reform, name="C11")
model.addConstr(RG_H == yield_reform_gas_heavy_naphtha * N_H_to_reform, name="C12")

# C13–C14: Light/heavy oil allocation
model.addConstr(LO_to_mix + LO_to_crack == LO, name="C13")
model.addConstr(HO_to_mix + HO_to_crack == HO, name="C14")

# C15–C18: Cracking yields
model.addConstr(CO_LO == yield_pyrolysis_oil_light * LO_to_crack, name="C15")
model.addConstr(CG_LO == yield_pyrolysis_gasoline_light * LO_to_crack, name="C16")
model.addConstr(CO_HO == yield_pyrolysis_oil_heavy * HO_to_crack, name="C17")
model.addConstr(CG_HO == yield_pyrolysis_gasoline_heavy * HO_to_crack, name="C18")

# C19–C20: Total cracking outputs
model.addConstr(CO == CO_LO + CO_HO, name="C19")
model.addConstr(CG == CG_LO + CG_HO, name="C20")

# C21–C22: Residue allocation and lub
model.addConstr(R_to_mix + R_to_lub == R, name="C21")
model.addConstr(Lub == yield_lubricating_from_residual * R_to_lub, name="C22")

# C23–C29: Engine-oil blending allocation
model.addConstr(N_L_H + N_L_O == N_L_to_mix, name="C23")
model.addConstr(N_M_H + N_M_O == N_M_to_mix, name="C24")
model.addConstr(N_H_H + N_H_O == N_H_to_mix, name="C25")
model.addConstr(RG_L_H + RG_L_O == RG_L, name="C26")
model.addConstr(RG_M_H + RG_M_O == RG_M, name="C27")
model.addConstr(RG_H_H + RG_H_O == RG_H, name="C28")
model.addConstr(CG_H + CG_O == CG, name="C29")

# C30–C31: Engine-oil production balances
model.addConstr(
    HE == N_L_H + N_M_H + N_H_H + RG_L_H + RG_M_H + RG_H_H + CG_H,
    name="C30"
)
model.addConstr(
    OE == N_L_O + N_M_O + N_H_O + RG_L_O + RG_M_O + RG_H_O + CG_O,
    name="C31"
)

# C32: Octane constraint for high-end engine oil
model.addConstr(
    octane_naphtha['light'] * N_L_H
    + octane_naphtha['medium'] * N_M_H
    + octane_naphtha['heavy'] * N_H_H
    + octane_reformed_gasoline * (RG_L_H + RG_M_H + RG_H_H)
    + octane_cracking_gasoline * CG_H
    >= min_octane_premium_engine_oil * HE,
    name="C32"
)

# C33: Octane constraint for ordinary engine oil
model.addConstr(
    octane_naphtha['light'] * N_L_O
    + octane_naphtha['medium'] * N_M_O
    + octane_naphtha['heavy'] * N_H_O
    + octane_reformed_gasoline * (RG_L_O + RG_M_O + RG_H_O)
    + octane_cracking_gasoline * CG_O
    >= min_octane_ordinary_engine_oil * OE,
    name="C33"
)

# C34: Mixture gas-pressure constraint for kerosene/fuel blend pool
model.addConstr(
    pressure_light_oil * LO_to_mix
    + pressure_heavy_oil * HO_to_mix
    + pressure_pyrolysis_oil * CO
    + pressure_residual_oil * R_to_mix
    <= LO_to_mix + HO_to_mix + CO + R_to_mix,
    name="C34"
)

# Fuel oil composition ratio using additional variables (instead of C35–C37 directly)
# FO components must be in fixed ratio 10:3:4:1 via lambda_FO

model.addConstr(FO_LO == ratio_fuel_oil['light_oil'] * lambda_FO, name="FO_ratio_LO")
model.addConstr(FO_HO == ratio_fuel_oil['heavy_oil'] * lambda_FO, name="FO_ratio_HO")
model.addConstr(FO_CO == ratio_fuel_oil['cracking_oil'] * lambda_FO, name="FO_ratio_CO")
model.addConstr(FO_R == ratio_fuel_oil['residual_oil'] * lambda_FO, name="FO_ratio_R")

# FO amount is sum of components
model.addConstr(
    FO == FO_LO + FO_HO + FO_CO + FO_R,
    name="FO_total"
)

# Component availability limits
model.addConstr(FO_LO <= LO_to_mix, name="FO_LO_avail")
model.addConstr(FO_HO <= HO_to_mix, name="FO_HO_avail")
model.addConstr(FO_CO <= CO, name="FO_CO_avail")
model.addConstr(FO_R <= R_to_mix, name="FO_R_avail")

# Kerosene is the remaining part of the mix pool not used in FO
model.addConstr(
    K == (LO_to_mix - FO_LO)
    + (HO_to_mix - FO_HO)
    + (CO - FO_CO)
    + (R_to_mix - FO_R),
    name="K_balance"
)

# C35–C37 from the validated model are NOT enforced explicitly, because
# the extended FO-ratio modeling with lambda_FO and FO_* is used instead.

# C38–C39: Crude availability (already via ub, but add explicit constraints)
model.addConstr(D1 <= avail_crude_oil_1, name="C38")
model.addConstr(D2 <= avail_crude_oil_2, name="C39")

# C40: Distillation capacity
model.addConstr(D1 + D2 <= cap_distillation, name="C40")

# C41: Reforming capacity
model.addConstr(
    N_L_to_reform + N_M_to_reform + N_H_to_reform <= cap_reforming,
    name="C41"
)

# C42: Cracking capacity
model.addConstr(
    LO_to_crack + HO_to_crack <= cap_cracking,
    name="C42"
)

# C43: Lubricating oil bounds
model.addConstr(Lub >= lubricating_oil_min, name="C43_min")
model.addConstr(Lub <= lubricating_oil_max, name="C43_max")

# C44: Minimum ratio of HE to OE
model.addConstr(HE >= min_ratio_premium_to_ordinary * OE, name="C44")

# Nonnegativity is already enforced via lb=0.0

# ==========================
# 5. Objective function
# ==========================

obj = (
    profit_premium_engine_oil * HE
    + profit_ordinary_engine_oil * OE
    + profit_kerosene * K
    + profit_fuel_oil * FO
    + profit_lubricating_oil * Lub
)

model.setObjective(obj, GRB.MAXIMIZE)

# ==========================
# 6. Optimize
# ==========================

model.Params.OutputFlag = 0  # Turn off solver output for cleanliness; remove if desired
model.optimize()

# ==========================
# 7. Print results
# ==========================

if model.status == GRB.OPTIMAL:
    max_profit = model.objVal
    print(f"Optimal total profit: {max_profit:.2f}")
else:
    max_profit = float('nan')
    print("No optimal solution found.")

# Required final statement:
print(f"FinalAnswer=【{max_profit}】")