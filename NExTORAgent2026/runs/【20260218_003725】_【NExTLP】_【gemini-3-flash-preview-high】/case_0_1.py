import gurobipy as gp
from gurobipy import GRB

def solve_refinery_problem():
    # Create the model
    model = gp.Model("Refinery_Production_Flow")

    # Parameters List
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

    # Decision Variables
    D1 = model.addVar(lb=0, ub=avail_crude_oil_1, name="D1")
    D2 = model.addVar(lb=0, ub=avail_crude_oil_2, name="D2")
    N_L = model.addVar(lb=0, name="N_L")
    N_M = model.addVar(lb=0, name="N_M")
    N_H = model.addVar(lb=0, name="N_H")
    LO = model.addVar(lb=0, name="LO")
    HO = model.addVar(lb=0, name="HO")
    R = model.addVar(lb=0, name="R")
    N_L_to_mix = model.addVar(lb=0, name="N_L_to_mix")
    N_L_to_reform = model.addVar(lb=0, name="N_L_to_reform")
    N_M_to_mix = model.addVar(lb=0, name="N_M_to_mix")
    N_M_to_reform = model.addVar(lb=0, name="N_M_to_reform")
    N_H_to_mix = model.addVar(lb=0, name="N_H_to_mix")
    N_H_to_reform = model.addVar(lb=0, name="N_H_to_reform")
    RG_L = model.addVar(lb=0, name="RG_L")
    RG_M = model.addVar(lb=0, name="RG_M")
    RG_H = model.addVar(lb=0, name="RG_H")
    LO_to_mix = model.addVar(lb=0, name="LO_to_mix")
    LO_to_crack = model.addVar(lb=0, name="LO_to_crack")
    HO_to_mix = model.addVar(lb=0, name="HO_to_mix")
    HO_to_crack = model.addVar(lb=0, name="HO_to_crack")
    CO_LO = model.addVar(lb=0, name="CO_LO")
    CG_LO = model.addVar(lb=0, name="CG_LO")
    CO_HO = model.addVar(lb=0, name="CO_HO")
    CG_HO = model.addVar(lb=0, name="CG_HO")
    CO = model.addVar(lb=0, name="CO")
    CG = model.addVar(lb=0, name="CG")
    R_to_mix = model.addVar(lb=0, name="R_to_mix")
    R_to_lub = model.addVar(lb=0, name="R_to_lub")
    Lub = model.addVar(lb=lubricating_oil_min, ub=lubricating_oil_max, name="Lub")
    N_L_H = model.addVar(lb=0, name="N_L_H")
    N_M_H = model.addVar(lb=0, name="N_M_H")
    N_H_H = model.addVar(lb=0, name="N_H_H")
    RG_L_H = model.addVar(lb=0, name="RG_L_H")
    RG_M_H = model.addVar(lb=0, name="RG_M_H")
    RG_H_H = model.addVar(lb=0, name="RG_H_H")
    CG_H = model.addVar(lb=0, name="CG_H")
    N_L_O = model.addVar(lb=0, name="N_L_O")
    N_M_O = model.addVar(lb=0, name="N_M_O")
    N_H_O = model.addVar(lb=0, name="N_H_O")
    RG_L_O = model.addVar(lb=0, name="RG_L_O")
    RG_M_O = model.addVar(lb=0, name="RG_M_O")
    RG_H_O = model.addVar(lb=0, name="RG_H_O")
    CG_O = model.addVar(lb=0, name="CG_O")
    HE = model.addVar(lb=0, name="HE")
    OE = model.addVar(lb=0, name="OE")
    K = model.addVar(lb=0, name="K")
    FO = model.addVar(lb=0, name="FO")

    # Objective Function
    model.setObjective(profit_premium_engine_oil * HE + profit_ordinary_engine_oil * OE + 
                       profit_kerosene * K + profit_fuel_oil * FO + 
                       profit_lubricating_oil * Lub, GRB.MAXIMIZE)

    # Constraints
    model.addConstr(N_L == Table_1_C_1['Crude oil 1']['Light naphtha'] * D1 + Table_1_C_1['Crude oil 2']['Light naphtha'] * D2, "C1")
    model.addConstr(N_M == Table_1_C_1['Crude oil 1']['Medium naphtha'] * D1 + Table_1_C_1['Crude oil 2']['Medium naphtha'] * D2, "C2")
    model.addConstr(N_H == Table_1_C_1['Crude oil 1']['Heavy naphtha'] * D1 + Table_1_C_1['Crude oil 2']['Heavy naphtha'] * D2, "C3")
    model.addConstr(LO == Table_1_C_1['Crude oil 1']['Light oil'] * D1 + Table_1_C_1['Crude oil 2']['Light oil'] * D2, "C4")
    model.addConstr(HO == Table_1_C_1['Crude oil 1']['Heavy oil'] * D1 + Table_1_C_1['Crude oil 2']['Heavy oil'] * D2, "C5")
    model.addConstr(R == Table_1_C_1['Crude oil 1']['Residue'] * D1 + Table_1_C_1['Crude oil 2']['Residue'] * D2, "C6")

    model.addConstr(N_L_to_mix + N_L_to_reform == N_L, "C7")
    model.addConstr(N_M_to_mix + N_M_to_reform == N_M, "C8")
    model.addConstr(N_H_to_mix + N_H_to_reform == N_H, "C9")

    model.addConstr(RG_L == yield_reform_gas_light_naphtha * N_L_to_reform, "C10")
    model.addConstr(RG_M == yield_reform_gas_medium_naphtha * N_M_to_reform, "C11")
    model.addConstr(RG_H == yield_reform_gas_heavy_naphtha * N_H_to_reform, "C12")

    model.addConstr(LO_to_mix + LO_to_crack == LO, "C13")
    model.addConstr(HO_to_mix + HO_to_crack == HO, "C14")

    model.addConstr(CO_LO == yield_pyrolysis_oil_light * LO_to_crack, "C15")
    model.addConstr(CG_LO == yield_pyrolysis_gasoline_light * LO_to_crack, "C16")
    model.addConstr(CO_HO == yield_pyrolysis_oil_heavy * HO_to_crack, "C17")
    model.addConstr(CG_HO == yield_pyrolysis_gasoline_heavy * HO_to_crack, "C18")
    model.addConstr(CO == CO_LO + CO_HO, "C19")
    model.addConstr(CG == CG_LO + CG_HO, "C20")

    model.addConstr(R_to_mix + R_to_lub == R, "C21")
    model.addConstr(Lub == yield_lubricating_from_residual * R_to_lub, "C22")

    model.addConstr(N_L_H + N_L_O == N_L_to_mix, "C23")
    model.addConstr(N_M_H + N_M_O == N_M_to_mix, "C24")
    model.addConstr(N_H_H + N_H_O == N_H_to_mix, "C25")
    model.addConstr(RG_L_H + RG_L_O == RG_L, "C26")
    model.addConstr(RG_M_H + RG_M_O == RG_M, "C27")
    model.addConstr(RG_H_H + RG_H_O == RG_H, "C28")
    model.addConstr(CG_H + CG_O == CG, "C29")

    model.addConstr(HE == N_L_H + N_M_H + N_H_H + RG_L_H + RG_M_H + RG_H_H + CG_H, "C30")
    model.addConstr(OE == N_L_O + N_M_O + N_H_O + RG_L_O + RG_M_O + RG_H_O + CG_O, "C31")

    model.addConstr(octane_naphtha['light'] * N_L_H + octane_naphtha['medium'] * N_M_H + octane_naphtha['heavy'] * N_H_H + 
                    octane_reformed_gasoline * (RG_L_H + RG_M_H + RG_H_H) + octane_cracking_gasoline * CG_H >= min_octane_premium_engine_oil * HE, "C32")
    model.addConstr(octane_naphtha['light'] * N_L_O + octane_naphtha['medium'] * N_M_O + octane_naphtha['heavy'] * N_H_O + 
                    octane_reformed_gasoline * (RG_L_O + RG_M_O + RG_H_O) + octane_cracking_gasoline * CG_O >= min_octane_ordinary_engine_oil * OE, "C33")

    model.addConstr(pressure_light_oil * LO_to_mix + pressure_heavy_oil * HO_to_mix + pressure_pyrolysis_oil * CO + pressure_residual_oil * R_to_mix <= max_pressure_kerosene * (K + FO), "C34")

    # Pool ratios for P leading to K and FO
    model.addConstr(ratio_fuel_oil['heavy_oil'] * LO_to_mix == ratio_fuel_oil['light_oil'] * HO_to_mix, "C35")
    model.addConstr(ratio_fuel_oil['cracking_oil'] * HO_to_mix == ratio_fuel_oil['heavy_oil'] * CO, "C36")
    model.addConstr(ratio_fuel_oil['residual_oil'] * LO_to_mix == ratio_fuel_oil['light_oil'] * R_to_mix, "C37")
    model.addConstr(K + FO == LO_to_mix + HO_to_mix + CO + R_to_mix, "BalancePool")

    model.addConstr(D1 + D2 <= cap_distillation, "C40")
    model.addConstr(N_L_to_reform + N_M_to_reform + N_H_to_reform <= cap_reforming, "C41")
    model.addConstr(LO_to_crack + HO_to_crack <= cap_cracking, "C42")
    model.addConstr(HE >= min_ratio_premium_to_ordinary * OE, "C44")

    # Solve the model
    model.optimize()

    # Print FinalAnswer
    if model.status == GRB.OPTIMAL:
        print(f"FinalAnswer=【{model.objVal}】")
    else:
        print("No optimal solution found.")

if __name__ == "__main__":
    solve_refinery_problem()