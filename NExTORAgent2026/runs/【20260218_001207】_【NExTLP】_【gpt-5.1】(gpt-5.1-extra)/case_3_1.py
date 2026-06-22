import gurobipy as gp
from gurobipy import GRB

# ============================================
# 1. Import parameters directly from the given Parameters List
#    (values are taken exactly as provided)
# ============================================

Company_Name = 'Ethiopian Adama Chemical Company'
Company_Abbreviation = 'EAC'
Establishment_Year = 1998
Distance_to_Modjo_Dry_Port_km = 72
Planned_PV_Capacity_MW = 5
Planned_PV_Operation_Year = 2026
Planning_Year = 2025
Planning_Horizon_Quarters = 2
Rainy_Season_Months = ['June', 'July', 'August', 'September']
Products = ['Fertilizer', 'Paint', 'Chemicals']
Product_Codes = {'Fertilizer': 'F', 'Paint': 'P', 'Chemicals': 'C'}
Number_of_Suppliers = 10
Max_Supply_per_Supplier_tons = 500
Total_Max_Raw_Material_Supply_tons = 5000
Max_Average_Raw_Material_per_ton_tons = 0.65
Max_Reactor_Operating_Days = 125
Shifts_per_Day = 2
Hours_per_Shift = 40
Max_Production_per_Product_tons = 3000
Number_of_Operators = 80
Max_Net_Working_Hours_per_Operator = 100
Total_Available_Labor_Hours = 8000
Max_Demand_Fertilizer_tons = 2000
Max_Demand_Paint_tons = 1500
Max_Demand_Chemicals_tons = 1800
Min_Fertilizer_Share = 0.25
Min_Paint_Production_tons = 1000
Chemicals_to_Fertilizer_Ratio_Max = 0.8
Donation_Ratio = 0.01
Profit_Coefficients_Already_Net_of_Donation = True
Profit_per_ton_USD = {'Fertilizer': 200, 'Paint': 300, 'Chemicals': 250}
Raw_materials_per_ton_tons = {'Fertilizer': 0.5, 'Paint': 0.7, 'Chemicals': 0.6}
Machine_time_per_ton_hours = {'Fertilizer': 0.8, 'Paint': 1.0, 'Chemicals': 0.9}
Labor_per_ton_hours = {'Fertilizer': 0.6, 'Paint': 0.8, 'Chemicals': 0.7}
Table_1_CostData = [
    {
        'Product': 'Fertilizer',
        'Profit_per_ton_USD': 200,
        'Raw_materials_per_ton_tons': 0.5,
        'Machine_time_per_ton_hours': 0.8,
        'Labor_per_ton_hours': 0.6
    },
    {
        'Product': 'Paint',
        'Profit_per_ton_USD': 300,
        'Raw_materials_per_ton_tons': 0.7,
        'Machine_time_per_ton_hours': 1.0,
        'Labor_per_ton_hours': 0.8
    },
    {
        'Product': 'Chemicals',
        'Profit_per_ton_USD': 250,
        'Raw_materials_per_ton_tons': 0.6,
        'Machine_time_per_ton_hours': 0.9,
        'Labor_per_ton_hours': 0.7
    }
]

# Derived capacity values from parameters
Total_Machine_Time_Hours = Max_Reactor_Operating_Days * Shifts_per_Day * Hours_per_Shift

# ============================================
# 2. Create Gurobi model
# ============================================

model = gp.Model("EAC_Production_Planning")

# ============================================
# 3. Create decision variables
# ============================================

# Production quantities (tons)
x_F = model.addVar(lb=0.0, name="x_F")  # Fertilizer
x_P = model.addVar(lb=0.0, name="x_P")  # Paint
x_C = model.addVar(lb=0.0, name="x_C")  # Chemicals

# ============================================
# 4. Auxiliary substitution / indicator variables
#    Not required here: final model is linear.
# ============================================
# (If any were needed, they would be declared with lb=-GRB.INFINITY, ub=GRB.INFINITY.)

# ============================================
# 5. Objective function
#    Maximize Z = 200 x_F + 300 x_P + 250 x_C
# ============================================

model.setObjective(
    Profit_per_ton_USD['Fertilizer'] * x_F +
    Profit_per_ton_USD['Paint'] * x_P +
    Profit_per_ton_USD['Chemicals'] * x_C,
    GRB.MAXIMIZE
)

# ============================================
# 6. Add constraints
# ============================================

# 6.1 Raw material availability:
# 0.5 x_F + 0.7 x_P + 0.6 x_C ≤ 5000
model.addConstr(
    Raw_materials_per_ton_tons['Fertilizer'] * x_F +
    Raw_materials_per_ton_tons['Paint'] * x_P +
    Raw_materials_per_ton_tons['Chemicals'] * x_C
    <= Total_Max_Raw_Material_Supply_tons,
    name="RawMaterialCapacity"
)

# 6.2 Machine time capacity:
# 0.8 x_F + 1.0 x_P + 0.9 x_C ≤ 10000
model.addConstr(
    Machine_time_per_ton_hours['Fertilizer'] * x_F +
    Machine_time_per_ton_hours['Paint'] * x_P +
    Machine_time_per_ton_hours['Chemicals'] * x_C
    <= Total_Machine_Time_Hours,
    name="MachineTimeCapacity"
)

# 6.3 Labor time capacity:
# 0.6 x_F + 0.8 x_P + 0.7 x_C ≤ 8000
model.addConstr(
    Labor_per_ton_hours['Fertilizer'] * x_F +
    Labor_per_ton_hours['Paint'] * x_P +
    Labor_per_ton_hours['Chemicals'] * x_C
    <= Total_Available_Labor_Hours,
    name="LaborCapacity"
)

# 6.4 Per-product production capacity (equipment) and demand
# Capacity: each product ≤ 3000 tons
model.addConstr(x_F <= Max_Production_per_Product_tons, name="Cap_Fertilizer")
model.addConstr(x_P <= Max_Production_per_Product_tons, name="Cap_Paint")
model.addConstr(x_C <= Max_Production_per_Product_tons, name="Cap_Chemicals")

# Demand upper bounds: Fertilizer ≤ 2000, Paint ≤ 1500, Chemicals ≤ 1800
model.addConstr(x_F <= Max_Demand_Fertilizer_tons, name="Demand_Fertilizer")
model.addConstr(x_P <= Max_Demand_Paint_tons, name="Demand_Paint")
model.addConstr(x_C <= Max_Demand_Chemicals_tons, name="Demand_Chemicals")

# 6.5 Fertilizer minimum share of total output:
# x_F ≥ 0.25 (x_F + x_P + x_C) ⇔ 3 x_F ≥ x_P + x_C
model.addConstr(
    3 * x_F >= x_P + x_C,
    name="MinFertilizerShare"
)

# 6.6 Minimum paint output:
# x_P ≥ 1000
model.addConstr(
    x_P >= Min_Paint_Production_tons,
    name="MinPaintProduction"
)

# 6.7 Chemicals limited by fertilizer output:
# x_C ≤ 0.8 x_F
model.addConstr(
    x_C <= Chemicals_to_Fertilizer_Ratio_Max * x_F,
    name="ChemicalsFertilizerRatio"
)

# 6.8 Average raw material consumption per ton (linearized):
# 0.5 x_F + 0.7 x_P + 0.6 x_C ≤ 0.65 (x_F + x_P + x_C)
# ⇔ 0.7 x_P + 0.6 x_C - 0.15 x_F ≤ 0
model.addConstr(
    0.7 * x_P + 0.6 * x_C - 0.15 * x_F <= 0,
    name="AvgRawMaterialPerTon"
)

# Non-negativity is already enforced via lb=0 on variables.

# ============================================
# 7. Solve the model and print results
# ============================================

model.optimize()

if model.status == GRB.OPTIMAL:
    xF_val = x_F.X
    xP_val = x_P.X
    xC_val = x_C.X
    optimal_profit = model.ObjVal

    # Optional detailed output
    print("Optimal production plan:")
    print(f"  Fertilizer (x_F): {xF_val:.4f} tons")
    print(f"  Paint      (x_P): {xP_val:.4f} tons")
    print(f"  Chemicals  (x_C): {xC_val:.4f} tons")
    print(f"Maximum total profit (USD): {optimal_profit:.2f}")

    # Final required answer: only the maximum total profit
    print(f"FinalAnswer=【{optimal_profit:.2f}】")
else:
    print("Model did not solve to optimality.")
    print("FinalAnswer=【NaN】")