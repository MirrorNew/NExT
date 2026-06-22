import gurobipy as gp
from gurobipy import GRB

def solve_agricultural_problem():
    # Initialize Model
    model = gp.Model("Agricultural_Optimization")

    # -------------------------------------------------------------------------
    # 1. Define Parameters (strictly from Parameters List)
    # -------------------------------------------------------------------------
    Total_Arable_Land_Area = 100
    Min_Planting_Area_Per_Crop = 10
    Min_Planting_Area_Wheat = 20
    Max_Planting_Area_Cotton = 30
    Max_Planting_Area_Corn_and_Cotton = 80
    
    Water_Limit = 500
    # Threshold is 80% of total supply
    Water_Consumption_Threshold_Ratio = 0.8 
    Water_Threshold = Water_Limit * Water_Consumption_Threshold_Ratio # 400
    
    Labor_Per_Unit_Irrigation_Water_Over = 2
    
    Fertilizer_Supply_Base = 150
    Max_Additional_Fertilizer = 20
    # Total Fertilizer limit = Base + Max Purchase
    Fertilizer_Limit = Fertilizer_Supply_Base + Max_Additional_Fertilizer # 170
    
    Labor_Per_Additional_Fertilizer_Unit = 2
    
    Labor_Capacity_Per_Worker = 2
    Total_Number_of_Workers = 500
    Total_Labor_Supply_Hours = Total_Number_of_Workers * Labor_Capacity_Per_Worker # 1000

    # Crop Data: [Profit, Labor, Water, Fertilizer]
    # Structure based on 'Table_1_Crop_Data'
    crop_data = {
        'Wheat':   {'Profit': 300, 'Labor': 10, 'Water': 5, 'Fert': 2},
        'Corn':    {'Profit': 400, 'Labor': 8,  'Water': 7, 'Fert': 3},
        'Soybean': {'Profit': 250, 'Labor': 5,  'Water': 4, 'Fert': 1},
        'Cotton':  {'Profit': 500, 'Labor': 12, 'Water': 9, 'Fert': 4}
    }
    crops = list(crop_data.keys())

    # -------------------------------------------------------------------------
    # 2. Decision Variables
    # -------------------------------------------------------------------------
    # Planting area for each crop (Continuous)
    x = model.addVars(crops, lb=0, ub=Total_Arable_Land_Area, vtype=GRB.CONTINUOUS, name="x")

    # Workers assigned to each crop (Integer)
    # Note: The problem asks to "choose how many workers to send". 
    # While labor *demand* is calculated in man-hours, the assignment is persons.
    w = model.addVars(crops, lb=0, ub=Total_Number_of_Workers, vtype=GRB.INTEGER, name="w")

    # Resource usage variables (Continuous)
    W = model.addVar(lb=0, ub=Water_Limit, vtype=GRB.CONTINUOUS, name="W_TotalWater")
    F = model.addVar(lb=0, ub=Fertilizer_Limit, vtype=GRB.CONTINUOUS, name="F_TotalFertilizer")

    # Auxiliary variables for extra labor (Linearization)
    lab_w = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="lab_w_WaterExtra")
    lab_f = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="lab_f_FertExtra")

    # -------------------------------------------------------------------------
    # 3. Objective Function
    # -------------------------------------------------------------------------
    # Maximize total profit
    model.setObjective(
        gp.quicksum(crop_data[c]['Profit'] * x[c] for c in crops), 
        GRB.MAXIMIZE
    )

    # -------------------------------------------------------------------------
    # 4. Constraints
    # -------------------------------------------------------------------------
    
    # --- Land Constraints ---
    # Total land usage must equal available land
    model.addConstr(gp.quicksum(x[c] for c in crops) == Total_Arable_Land_Area, "TotalLandUsage")
    
    # Minimum area per crop
    for c in crops:
        if c == 'Wheat':
            model.addConstr(x[c] >= Min_Planting_Area_Wheat, f"MinArea_{c}")
        else:
            model.addConstr(x[c] >= Min_Planting_Area_Per_Crop, f"MinArea_{c}")
    
    # Maximum cotton area
    model.addConstr(x['Cotton'] <= Max_Planting_Area_Cotton, "MaxCottonArea")
    
    # Combined Corn and Cotton area limit
    model.addConstr(x['Corn'] + x['Cotton'] <= Max_Planting_Area_Corn_and_Cotton, "MaxCornCottonArea")

    # --- Water Constraints ---
    # Define Total Water Usage
    model.addConstr(W == gp.quicksum(crop_data[c]['Water'] * x[c] for c in crops), "WaterDef")
    
    # Water Usage Limit (Explicit)
    model.addConstr(W <= Water_Limit, "WaterLimit")
    
    # Water Transport Labor Linearization: lab_w >= 2 * (W - 400)
    # Since we minimize impact on labor capacity to allow max profit, simple inequalities work.
    model.addConstr(lab_w >= Labor_Per_Unit_Irrigation_Water_Over * (W - Water_Threshold), "WaterLaborLin1")
    # lab_w >= 0 is handled by variable bounds.

    # --- Fertilizer Constraints ---
    # Define Total Fertilizer Usage
    model.addConstr(F == gp.quicksum(crop_data[c]['Fert'] * x[c] for c in crops), "FertDef")
    
    # Fertilizer Usage Limit (Explicit)
    model.addConstr(F <= Fertilizer_Limit, "FertLimit")
    
    # Fertilizer Purchase Labor Linearization: lab_f >= 2 * (F - 150)
    model.addConstr(lab_f >= Labor_Per_Additional_Fertilizer_Unit * (F - Fertilizer_Supply_Base), "FertLaborLin1")
    # lab_f >= 0 is handled by variable bounds.

    # --- Labor Constraints ---
    # Total assigned workers cannot exceed total workers
    model.addConstr(gp.quicksum(w[c] for c in crops) <= Total_Number_of_Workers, "TotalWorkersLimit")
    
    # Labor Supply vs Demand
    # Supply: 2 hours * Total Workers Assigned
    # Demand: Crop Planting Labor + Water Extra Labor + Fertilizer Extra Labor
    planting_labor_demand = gp.quicksum(crop_data[c]['Labor'] * x[c] for c in crops)
    assigned_labor_supply = Labor_Capacity_Per_Worker * gp.quicksum(w[c] for c in crops)
    
    model.addConstr(assigned_labor_supply >= planting_labor_demand + lab_w + lab_f, "LaborBalance")

    # -------------------------------------------------------------------------
    # 5. Solve and Output
    # -------------------------------------------------------------------------
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        # The question asks for the "maximum profit value"
        print(f"FinalAnswer=【{model.ObjVal}】")
    else:
        print("FinalAnswer=【NoFeasibleSolution】")

# Execute the function
solve_agricultural_problem()