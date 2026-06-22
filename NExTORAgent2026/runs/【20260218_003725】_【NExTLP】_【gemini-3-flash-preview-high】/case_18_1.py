import gurobipy as gp
from gurobipy import GRB

# Create the model
model = gp.Model("Agriculture_Optimization")

# Parameters
# Table 1: Crop Data (Profit, Labor Demand, Water Demand, Fertilizer Demand)
# Table 2: Resource Supply (Workers, Max Water, Available Fertilizer, Land Area)

# Decision Variables
x_w = model.addVar(lb=20, ub=100, name="x_w")  # Wheat area, minimum 20 hectares
x_c = model.addVar(lb=10, ub=100, name="x_c")  # Corn area, minimum 10 hectares
x_s = model.addVar(lb=10, ub=100, name="x_s")  # Soybean area, minimum 10 hectares
x_co = model.addVar(lb=10, ub=30, name="x_co") # Cotton area, min 10, max 30 hectares

# Workers assigned to each crop (integer)
w_w = model.addVar(vtype=GRB.INTEGER, lb=0, ub=500, name="w_w")
w_c = model.addVar(vtype=GRB.INTEGER, lb=0, ub=500, name="w_c")
w_s = model.addVar(vtype=GRB.INTEGER, lb=0, ub=500, name="w_s")
w_co = model.addVar(vtype=GRB.INTEGER, lb=0, ub=500, name="w_co")

# Resource usage variables
W = model.addVar(lb=0, ub=500, name="W")  # Total irrigation water (thousand m3)
F = model.addVar(lb=0, ub=170, name="F")  # Total fertilizer (tons, max 150 + 20)

# Additional labor for resource transport
lab_w = model.addVar(lb=0, ub=200, name="lab_w") # Man-hours for additional water
lab_f = model.addVar(lb=0, ub=40, name="lab_f")  # Man-hours for additional fertilizer

# Indicator variables for thresholds
y_w = model.addVar(vtype=GRB.BINARY, name="y_w") # Triggered if W > 400
y_f = model.addVar(vtype=GRB.BINARY, name="y_f") # Triggered if F > 150

# Objective Function: Maximize Economic Benefits (Profit)
# Profit: Wheat (300), Corn (400), Soybean (250), Cotton (500)
model.setObjective(300*x_w + 400*x_c + 250*x_s + 500*x_co, GRB.MAXIMIZE)

# Constraints

# 1. Total land usage: all 100 hectares must be used without waste
model.addConstr(x_w + x_c + x_s + x_co == 100, name="Total_Land_Usage")

# 2. Corn and Cotton total planting area limit
model.addConstr(x_c + x_co <= 80, name="Corn_Cotton_Limit")

# 3. Irrigation water definition and usage limit
model.addConstr(W == 5*x_w + 7*x_c + 4*x_s + 9*x_co, name="Water_Definition")

# 4. Fertilizer usage definition and maximum consumption
model.addConstr(F == 2*x_w + 3*x_c + 1*x_s + 4*x_co, name="Fertilizer_Definition")

# 5. Water transport labor (Indicator-based threshold at 80% of 500)
model.addGenConstrIndicator(y_w, 1, W >= 400)
model.addGenConstrIndicator(y_w, 0, W <= 400)
model.addGenConstrIndicator(y_w, 1, lab_w == 2 * (W - 400))
model.addGenConstrIndicator(y_w, 0, lab_w == 0)

# 6. Fertilizer transport labor (Indicator-based threshold at 150 tons)
model.addGenConstrIndicator(y_f, 1, F >= 150)
model.addGenConstrIndicator(y_f, 0, F <= 150)
model.addGenConstrIndicator(y_f, 1, lab_f == 2 * (F - 150))
model.addGenConstrIndicator(y_f, 0, lab_f == 0)

# 7. Labor supply and demand balance
# Planting labor: Wheat(10), Corn(8), Soybean(5), Cotton(12)
planting_labor = 10*x_w + 8*x_c + 5*x_s + 12*x_co
total_workers_assigned = w_w + w_c + w_s + w_co
# Total labor capacity (2 hours per worker) must cover planting and transport labor
model.addConstr(2 * total_workers_assigned >= planting_labor + lab_w + lab_f, name="Labor_Supply_Demand")

# 8. Total workers limit
model.addConstr(total_workers_assigned <= 500, name="Total_Workers_Limit")

# Solve the model
model.optimize()

# Output the result
if model.status == GRB.OPTIMAL:
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("FinalAnswer=【No optimal solution found】")