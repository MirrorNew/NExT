import gurobipy as gp
from gurobipy import GRB

# =========================
# 1. Parameters (from Parameters List)
# =========================

CRUDE_TYPES = ['Crude oil 1', 'Crude oil 2']
NAPHTHA_TYPES = ['Light naphtha', 'Medium naphtha', 'Heavy naphtha']

Octane_Naphtha = {'Light naphtha': 90, 'Medium naphtha': 80, 'Heavy naphtha': 70}
Octane_ReformedGasoline = 115
Reforming_Yield = {'Light naphtha': 0.6, 'Medium naphtha': 0.52, 'Heavy naphtha': 0.45}
Octane_CrackingGasoline = 105

Cracking_Yield_LightOil = {'Cracking oil': 0.68, 'Cracking gasoline': 0.28}
Cracking_Yield_HeavyOil = {'Cracking oil': 0.75, 'Cracking gasoline': 0.2}

Residue_to_LubeOil_Yield = 0.5
MinOctane_PremiumEngineOil = 94
MinOctane_OrdinaryEngineOil = 84

MaxVaporPressure_KeroseneBlend = 1.0
VaporPressure_Components = {
    'Light oil': 1.0,
    'Heavy oil': 0.6,
    'Cracking oil': 1.5,
    'Residue': 0.05
}

FuelOil_Ratio = {'Light oil': 10, 'Heavy oil': 3, 'Cracking oil': 4, 'Residue': 1}

Available_CrudeOil1 = 20000
Available_CrudeOil2 = 30000
MaxCapacity_Distillation = 45000
MaxCapacity_Reforming = 10000
MaxCapacity_Cracking = 8000
MinOutput_LubeOil = 500
MaxOutput_LubeOil = 1000
MinRatio_Premium_to_Ordinary_EngineOil = 0.4

Profit = {
    'Premium engine oil': 700,
    'Ordinary engine oil': 600,
    'Kerosene': 400,
    'Fuel oil': 350,
    'Lubricating oil': 150
}

Table_1_DistillationYields = {
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

# =========================
# 2. Create model
# =========================

model = gp.Model("Refinery_Production_Flow")

# =========================
# 3. Decision variables
# =========================

# Crude inputs
C1 = model.addVar(name="C1", lb=0, ub=Available_CrudeOil1)
C2 = model.addVar(name="C2", lb=0, ub=Available_CrudeOil2)

# Naphtha splits (crude 1)
LN1_dir = model.addVar(name="LN1_dir", lb=0)
LN1_ref = model.addVar(name="LN1_ref", lb=0)
MN1_dir = model.addVar(name="MN1_dir", lb=0)
MN1_ref = model.addVar(name="MN1_ref", lb=0)
HN1_dir = model.addVar(name="HN1_dir", lb=0)
HN1_ref = model.addVar(name="HN1_ref", lb=0)

# Naphtha splits (crude 2)
LN2_dir = model.addVar(name="LN2_dir", lb=0)
LN2_ref = model.addVar(name="LN2_ref", lb=0)
MN2_dir = model.addVar(name="MN2_dir", lb=0)
MN2_ref = model.addVar(name="MN2_ref", lb=0)
HN2_dir = model.addVar(name="HN2_dir", lb=0)
HN2_ref = model.addVar(name="HN2_ref", lb=0)

# Reforming products
RG_L = model.addVar(name="RG_L", lb=0)
RG_M = model.addVar(name="RG_M", lb=0)
RG_H = model.addVar(name="RG_H", lb=0)

# Light and heavy oil splits
LO1_dir = model.addVar(name="LO1_dir", lb=0)
LO1_cr = model.addVar(name="LO1_cr", lb=0)
LO2_dir = model.addVar(name="LO2_dir", lb=0)
LO2_cr = model.addVar(name="LO2_cr", lb=0)

HO1_dir = model.addVar(name="HO1_dir", lb=0)
HO1_cr = model.addVar(name="HO1_cr", lb=0)
HO2_dir = model.addVar(name="HO2_dir", lb=0)
HO2_cr = model.addVar(name="HO2_cr", lb=0)

# Cracking products
CG_LO = model.addVar(name="CG_LO", lb=0)
CO_LO = model.addVar(name="CO_LO", lb=0)
CG_HO = model.addVar(name="CG_HO", lb=0)
CO_HO = model.addVar(name="CO_HO", lb=0)
CG = model.addVar(name="CG", lb=0)
CO = model.addVar(name="CO", lb=0)

# Residue splits and lube
RES1_fuel = model.addVar(name="RES1_fuel", lb=0)
RES1_lube = model.addVar(name="RES1_lube", lb=0)
RES2_fuel = model.addVar(name="RES2_fuel", lb=0)
RES2_lube = model.addVar(name="RES2_lube", lb=0)
LUBE = model.addVar(name="LUBE", lb=MinOutput_LubeOil, ub=MaxOutput_LubeOil)

# Aggregated naphtha & reformate for blending
LN_dir = model.addVar(name="LN_dir", lb=0)
MN_dir = model.addVar(name="MN_dir", lb=0)
HN_dir = model.addVar(name="HN_dir", lb=0)
RG = model.addVar(name="RG", lb=0)

# Allocation to premium engine oil
x_LN_PH = model.addVar(name="x_LN_PH", lb=0)
x_MN_PH = model.addVar(name="x_MN_PH", lb=0)
x_HN_PH = model.addVar(name="x_HN_PH", lb=0)
x_RG_PH = model.addVar(name="x_RG_PH", lb=0)
x_CG_PH = model.addVar(name="x_CG_PH", lb=0)

# Allocation to ordinary engine oil
x_LN_PO = model.addVar(name="x_LN_PO", lb=0)
x_MN_PO = model.addVar(name="x_MN_PO", lb=0)
x_HN_PO = model.addVar(name="x_HN_PO", lb=0)
x_RG_PO = model.addVar(name="x_RG_PO", lb=0)
x_CG_PO = model.addVar(name="x_CG_PO", lb=0)

# Engine oil products
E_H = model.addVar(name="E_H", lb=0)
E_O = model.addVar(name="E_O", lb=0)

# Aggregated light/heavy oil & residue to blending
LO_dir = model.addVar(name="LO_dir", lb=0)
HO_dir = model.addVar(name="HO_dir", lb=0)
RES_fuel = model.addVar(name="RES_fuel", lb=0)

# Splits between kerosene and fuel oil
LO_K = model.addVar(name="LO_K", lb=0)
LO_F = model.addVar(name="LO_F", lb=0)
HO_K = model.addVar(name="HO_K", lb=0)
HO_F = model.addVar(name="HO_F", lb=0)
CO_K = model.addVar(name="CO_K", lb=0)
CO_F = model.addVar(name="CO_F", lb=0)
RES_K = model.addVar(name="RES_K", lb=0)
RES_F = model.addVar(name="RES_F", lb=0)

# Final products
K = model.addVar(name="K", lb=0)
F = model.addVar(name="F", lb=0)

# Fuel oil ratio scaling variable
t_F = model.addVar(name="t_F", lb=0)

# =========================
# 4. Constraints
# =========================

# Crude availability & distillation capacity
model.addConstr(C1 <= Available_CrudeOil1, name="Crude1_availability")
model.addConstr(C2 <= Available_CrudeOil2, name="Crude2_availability")
model.addConstr(C1 + C2 <= MaxCapacity_Distillation, name="Distillation_capacity")

# Naphtha balances (crude 1)
model.addConstr(LN1_dir + LN1_ref == Table_1_DistillationYields['Crude oil 1']['Light naphtha'] * C1,
                name="LN1_balance")
model.addConstr(MN1_dir + MN1_ref == Table_1_DistillationYields['Crude oil 1']['Medium naphtha'] * C1,
                name="MN1_balance")
model.addConstr(HN1_dir + HN1_ref == Table_1_DistillationYields['Crude oil 1']['Heavy naphtha'] * C1,
                name="HN1_balance")

# Naphtha balances (crude 2)
model.addConstr(LN2_dir + LN2_ref == Table_1_DistillationYields['Crude oil 2']['Light naphtha'] * C2,
                name="LN2_balance")
model.addConstr(MN2_dir + MN2_ref == Table_1_DistillationYields['Crude oil 2']['Medium naphtha'] * C2,
                name="MN2_balance")
model.addConstr(HN2_dir + HN2_ref == Table_1_DistillationYields['Crude oil 2']['Heavy naphtha'] * C2,
                name="HN2_balance")

# Reforming yields
model.addConstr(RG_L == Reforming_Yield['Light naphtha'] * (LN1_ref + LN2_ref),
                name="RG_L_yield")
model.addConstr(RG_M == Reforming_Yield['Medium naphtha'] * (MN1_ref + MN2_ref),
                name="RG_M_yield")
model.addConstr(RG_H == Reforming_Yield['Heavy naphtha'] * (HN1_ref + HN2_ref),
                name="RG_H_yield")

# Reforming capacity
model.addConstr(
    LN1_ref + LN2_ref + MN1_ref + MN2_ref + HN1_ref + HN2_ref <= MaxCapacity_Reforming,
    name="Reforming_capacity"
)

# Light oil balances
model.addConstr(LO1_dir + LO1_cr == Table_1_DistillationYields['Crude oil 1']['Light oil'] * C1,
                name="LO1_balance")
model.addConstr(LO2_dir + LO2_cr == Table_1_DistillationYields['Crude oil 2']['Light oil'] * C2,
                name="LO2_balance")

# Heavy oil balances
model.addConstr(HO1_dir + HO1_cr == Table_1_DistillationYields['Crude oil 1']['Heavy oil'] * C1,
                name="HO1_balance")
model.addConstr(HO2_dir + HO2_cr == Table_1_DistillationYields['Crude oil 2']['Heavy oil'] * C2,
                name="HO2_balance")

# Cracking capacity
model.addConstr(
    LO1_cr + LO2_cr + HO1_cr + HO2_cr <= MaxCapacity_Cracking,
    name="Cracking_capacity"
)

# Cracking yields (light oil)
model.addConstr(CG_LO == Cracking_Yield_LightOil['Cracking gasoline'] * (LO1_cr + LO2_cr),
                name="CG_LO_yield")
model.addConstr(CO_LO == Cracking_Yield_LightOil['Cracking oil'] * (LO1_cr + LO2_cr),
                name="CO_LO_yield")

# Cracking yields (heavy oil)
model.addConstr(CG_HO == Cracking_Yield_HeavyOil['Cracking gasoline'] * (HO1_cr + HO2_cr),
                name="CG_HO_yield")
model.addConstr(CO_HO == Cracking_Yield_HeavyOil['Cracking oil'] * (HO1_cr + HO2_cr),
                name="CO_HO_yield")

# Total cracking gasoline & oil
model.addConstr(CG == CG_LO + CG_HO, name="CG_total")
model.addConstr(CO == CO_LO + CO_HO, name="CO_total")

# Residue balances & lube
model.addConstr(RES1_fuel + RES1_lube == Table_1_DistillationYields['Crude oil 1']['Residue'] * C1,
                name="RES1_balance")
model.addConstr(RES2_fuel + RES2_lube == Table_1_DistillationYields['Crude oil 2']['Residue'] * C2,
                name="RES2_balance")

model.addConstr(LUBE == Residue_to_LubeOil_Yield * (RES1_lube + RES2_lube),
                name="LUBE_yield")

# Lube production bounds (already via lb/ub on LUBE, but add explicit)
model.addConstr(LUBE >= MinOutput_LubeOil, name="LUBE_min")
model.addConstr(LUBE <= MaxOutput_LubeOil, name="LUBE_max")

# Aggregated naphtha & reformate for blending
model.addConstr(LN_dir == LN1_dir + LN2_dir, name="LN_dir_def")
model.addConstr(MN_dir == MN1_dir + MN2_dir, name="MN_dir_def")
model.addConstr(HN_dir == HN1_dir + HN2_dir, name="HN_dir_def")
model.addConstr(RG == RG_L + RG_M + RG_H, name="RG_def")

# Allocation of naphtha & gasoline to premium/ordinary engine oils
model.addConstr(x_LN_PH + x_LN_PO == LN_dir, name="LN_alloc")
model.addConstr(x_MN_PH + x_MN_PO == MN_dir, name="MN_alloc")
model.addConstr(x_HN_PH + x_HN_PO == HN_dir, name="HN_alloc")
model.addConstr(x_RG_PH + x_RG_PO == RG, name="RG_alloc")
model.addConstr(x_CG_PH + x_CG_PO == CG, name="CG_alloc")

# Engine oil production definitions
model.addConstr(
    E_H == x_LN_PH + x_MN_PH + x_HN_PH + x_RG_PH + x_CG_PH,
    name="E_H_def"
)
model.addConstr(
    E_O == x_LN_PO + x_MN_PO + x_HN_PO + x_RG_PO + x_CG_PO,
    name="E_O_def"
)

# Premium vs ordinary relation
model.addConstr(
    E_H >= MinRatio_Premium_to_Ordinary_EngineOil * E_O,
    name="Premium_vs_Ordinary"
)

# Octane constraints
model.addConstr(
    Octane_Naphtha['Light naphtha'] * x_LN_PH +
    Octane_Naphtha['Medium naphtha'] * x_MN_PH +
    Octane_Naphtha['Heavy naphtha'] * x_HN_PH +
    Octane_ReformedGasoline * x_RG_PH +
    Octane_CrackingGasoline * x_CG_PH
    >= MinOctane_PremiumEngineOil * E_H,
    name="Premium_octane"
)

model.addConstr(
    Octane_Naphtha['Light naphtha'] * x_LN_PO +
    Octane_Naphtha['Medium naphtha'] * x_MN_PO +
    Octane_Naphtha['Heavy naphtha'] * x_HN_PO +
    Octane_ReformedGasoline * x_RG_PO +
    Octane_CrackingGasoline * x_CG_PO
    >= MinOctane_OrdinaryEngineOil * E_O,
    name="Ordinary_octane"
)

# Totals to kerosene/fuel blending
model.addConstr(LO_dir == LO1_dir + LO2_dir, name="LO_dir_def")
model.addConstr(HO_dir == HO1_dir + HO2_dir, name="HO_dir_def")
model.addConstr(RES_fuel == RES1_fuel + RES2_fuel, name="RES_fuel_def")

# Splits between kerosene and fuel
model.addConstr(LO_K + LO_F == LO_dir, name="LO_split")
model.addConstr(HO_K + HO_F == HO_dir, name="HO_split")
model.addConstr(CO_K + CO_F == CO, name="CO_split")
model.addConstr(RES_K + RES_F == RES_fuel, name="RES_split")

# Kerosene and fuel oil products definitions
model.addConstr(K == LO_K + HO_K + CO_K + RES_K, name="K_def")
model.addConstr(F == LO_F + HO_F + CO_F + RES_F, name="F_def")

# Kerosene vapor pressure constraint
model.addConstr(
    VaporPressure_Components['Light oil'] * LO_K +
    VaporPressure_Components['Heavy oil'] * HO_K +
    VaporPressure_Components['Cracking oil'] * CO_K +
    VaporPressure_Components['Residue'] * RES_K
    <= MaxVaporPressure_KeroseneBlend * K,
    name="Kerosene_vapor_pressure"
)

# Fuel oil ratio constraints (10:3:4:1)
model.addConstr(LO_F == FuelOil_Ratio['Light oil'] * t_F, name="Fuel_ratio_LO")
model.addConstr(HO_F == FuelOil_Ratio['Heavy oil'] * t_F, name="Fuel_ratio_HO")
model.addConstr(CO_F == FuelOil_Ratio['Cracking oil'] * t_F, name="Fuel_ratio_CO")
model.addConstr(RES_F == FuelOil_Ratio['Residue'] * t_F, name="Fuel_ratio_RES")

# Fuel oil production from ratio (F = 18 * t_F)
model.addConstr(F == (FuelOil_Ratio['Light oil'] +
                      FuelOil_Ratio['Heavy oil'] +
                      FuelOil_Ratio['Cracking oil'] +
                      FuelOil_Ratio['Residue']) * t_F,
                name="Fuel_ratio_F")

# =========================
# 5. Objective function
# =========================

obj = (
    Profit['Premium engine oil'] * E_H +
    Profit['Ordinary engine oil'] * E_O +
    Profit['Kerosene'] * K +
    Profit['Fuel oil'] * F +
    Profit['Lubricating oil'] * LUBE
)

model.setObjective(obj, GRB.MAXIMIZE)

# =========================
# 6. Solve model
# =========================

model.optimize()

# =========================
# 7. Print results
# =========================

if model.Status == GRB.OPTIMAL:
    optimal_profit = model.ObjVal
    print(f"Optimal total profit: {optimal_profit}")
    # FinalAnswer is the maximum value of total profit
    print(f"FinalAnswer=【{optimal_profit}】")
else:
    print("No optimal solution found.")
    # In case of no optimal solution, still output something for FinalAnswer
    print("FinalAnswer=【None】")