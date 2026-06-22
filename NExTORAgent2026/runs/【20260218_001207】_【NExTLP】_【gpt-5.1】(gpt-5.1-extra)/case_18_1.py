import gurobipy as gp
from gurobipy import GRB

# ==========================
# 1. Parameters (strictly from Parameters List)
# ==========================

CROPS = ['Wheat', 'Corn', 'Soybean', 'Cotton']

Total_arable_land_hectares = 100
Min_area_each_crop_hectares = 10
Min_area_wheat_hectares = 20
Max_area_cotton_hectares = 30
Max_area_corn_plus_cotton_hectares = 80

Water_trigger_fraction = 0.8          # given but not explicitly used in constraints
Water_remaining_fraction = 0.2        # given but not explicitly used in constraints

Labor_per_unit_additional_water = 2
Max_irrigation_water = 500

Labor_per_unit_additional_fertilizer = 2
Max_additional_fertilizer_tons = 20
Available_fertilizer_tons = 150       # looser than Max_additional_fertilizer_tons

Workers_count = 500
Labor_hours_per_worker = 2

Unit_profit_yuan_per_hectare = {
    'Wheat': 300,
    'Corn': 400,
    'Soybean': 250,
    'Cotton': 500
}

Labor_demand_man_hours_per_hectare = {
    'Wheat': 10,
    'Corn': 8,
    'Soybean': 5,
    'Cotton': 12
}

Irrigation_demand_thousand_cubic_meters_per_hectare = {
    'Wheat': 5,
    'Corn': 7,
    'Soybean': 4,
    'Cotton': 9
}

Fertilizer_demand_tons_per_hectare = {
    'Wheat': 2,
    'Corn': 3,
    'Soybean': 1,
    'Cotton': 4
}

Total_labor_hours_available = Workers_count * Labor_hours_per_worker

# ==========================
# 2. Create model
# ==========================

model = gp.Model("Agricultural_Production_Planning")

# ==========================
# 3. Decision variables
# ==========================

# Crop areas (hectares)
x = model.addVars(CROPS, name="Area", lb=0.0)

# Planting labor-hours per crop
L = model.addVars(CROPS, name="LaborPlanting", lb=0.0)

# Labor-hours for water and fertilizer logistics
L_water = model.addVar(name="Labor_Water", lb=0.0)
L_fert = model.addVar(name="Labor_Fertilizer", lb=0.0)

# Total irrigation water used (thousand cubic meters)
W = model.addVar(name="WaterUsed", lb=0.0)

# Total fertilizer used (tons)
F = model.addVar(name="FertilizerUsed", lb=0.0)

# ==========================
# 4. Objective function
# ==========================

model.setObjective(
    gp.quicksum(Unit_profit_yuan_per_hectare[crop] * x[crop] for crop in CROPS),
    GRB.MAXIMIZE
)

# ==========================
# 5. Constraints
# ==========================

# Land balance: all arable land used
model.addConstr(
    gp.quicksum(x[crop] for crop in CROPS) == Total_arable_land_hectares,
    name="LandBalance"
)

# Minimum area per crop
for crop in CROPS:
    model.addConstr(
        x[crop] >= Min_area_each_crop_hectares,
        name=f"MinArea_{crop}"
    )

# Minimum wheat area (stronger than generic minimum)
model.addConstr(
    x['Wheat'] >= Min_area_wheat_hectares,
    name="MinArea_Wheat_Stronger"
)

# Maximum cotton area
model.addConstr(
    x['Cotton'] <= Max_area_cotton_hectares,
    name="MaxArea_Cotton"
)

# Corn + cotton joint cap
model.addConstr(
    x['Corn'] + x['Cotton'] <= Max_area_corn_plus_cotton_hectares,
    name="CornPlusCottonCap"
)

# Water use definition from crop areas
model.addConstr(
    W == gp.quicksum(
        Irrigation_demand_thousand_cubic_meters_per_hectare[crop] * x[crop]
        for crop in CROPS
    ),
    name="WaterUseDefinition"
)

# Water availability limit
model.addConstr(
    W <= Max_irrigation_water,
    name="WaterLimit"
)

# Labor requirement for water transport: L_water = 2 * W
model.addConstr(
    L_water == Labor_per_unit_additional_water * W,
    name="LaborWaterDefinition"
)

# Fertilizer use definition from crop areas
model.addConstr(
    F == gp.quicksum(
        Fertilizer_demand_tons_per_hectare[crop] * x[crop]
        for crop in CROPS
    ),
    name="FertilizerUseDefinition"
)

# Fertilizer availability limit (tightest cap = Max_additional_fertilizer_tons = 20)
model.addConstr(
    F <= Max_additional_fertilizer_tons,
    name="FertilizerLimit"
)

# Labor requirement for fertilizer transport: L_fert = 2 * F
model.addConstr(
    L_fert == Labor_per_unit_additional_fertilizer * F,
    name="LaborFertilizerDefinition"
)

# Planting labor requirements per crop
for crop in CROPS:
    model.addConstr(
        L[crop] >= Labor_demand_man_hours_per_hectare[crop] * x[crop],
        name=f"PlantingLaborRequirement_{crop}"
    )

# Total labor-hours availability
model.addConstr(
    gp.quicksum(L[crop] for crop in CROPS) + L_water + L_fert
    <= Total_labor_hours_available,
    name="TotalLaborLimit"
)

# ==========================
# 6. Solve model
# ==========================

model.optimize()

# ==========================
# 7. Print results
# ==========================

if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
    print("Optimal solution found.")
    print(f"Maximum profit (objective value): {model.ObjVal:.4f} yuan\n")

    print("Crop areas (hectares):")
    for crop in CROPS:
        print(f"  {crop}: {x[crop].X:.4f}")

    print("\nPlanting labor-hours:")
    for crop in CROPS:
        print(f"  {crop}: {L[crop].X:.4f}")

    print(f"\nWater used (thousand m^3): {W.X:.4f}")
    print(f"Fertilizer used (tons): {F.X:.4f}")
    print(f"Labor for water transport (hours): {L_water.X:.4f}")
    print(f"Labor for fertilizer transport (hours): {L_fert.X:.4f}")
    total_labor_used = sum(L[c].X for c in CROPS) + L_water.X + L_fert.X
    print(f"Total labor used (hours): {total_labor_used:.4f}")

    # Final numeric answer for maximum profit
    print(f"FinalAnswer=【{model.ObjVal:.4f}】")
else:
    print("No optimal solution found.")
    print("FinalAnswer=【nan】")