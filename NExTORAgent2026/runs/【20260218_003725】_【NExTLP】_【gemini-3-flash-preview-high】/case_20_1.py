import gurobipy as gp
from gurobipy import GRB

# 1. Initialize the model
model = gp.Model("WeeklyMealPlanning")

# 2. Define data inputs and parameters
days = range(7)
packs = ['Pack A', 'Pack B', 'Pack C', 'Pack D']

# Nutrient and cost per 100g of individual ingredients
rice_nutrients = {'Cal': 360, 'Prot': 6, 'Fat': 1, 'VitC': 0, 'Cost': 0.5}
chicken_nutrients = {'Cal': 200, 'Prot': 20, 'Fat': 15, 'VitC': 0, 'Cost': 2.0}
beans_nutrients = {'Cal': 150, 'Prot': 8, 'Fat': 1, 'VitC': 5, 'Cost': 1.0}
milk_nutrients = {'Cal': 60, 'Prot': 3, 'Fat': 4, 'VitC': 5, 'Cost': 1.5}
veg_nutrients = {'Cal': 50, 'Prot': 2, 'Fat': 0, 'VitC': 20, 'Cost': 1.0}

# Pack compositions (in grams)
pack_comp = {
    'Pack A': {'Rice': 25, 'Chicken': 50, 'Beans': 0, 'Milk': 0},
    'Pack B': {'Rice': 25, 'Chicken': 40, 'Beans': 0, 'Milk': 0},
    'Pack C': {'Rice': 10, 'Chicken': 20, 'Beans': 20, 'Milk': 0},
    'Pack D': {'Rice': 0, 'Chicken': 0, 'Beans': 0, 'Milk': 50}
}

# Calculated coefficients for nutrients and cost per pack
pack_cal = {}
pack_prot = {}
pack_fat = {}
pack_vitC = {}
pack_chicken = {}
pack_beans = {}
pack_cost = {}

for p in packs:
    rice = pack_comp[p]['Rice']
    chicken = pack_comp[p]['Chicken']
    beans = pack_comp[p]['Beans']
    milk = pack_comp[p]['Milk']
    
    pack_cal[p] = (rice * rice_nutrients['Cal'] + chicken * chicken_nutrients['Cal'] + 
                   beans * beans_nutrients['Cal'] + milk * milk_nutrients['Cal']) / 100.0
    pack_prot[p] = (rice * rice_nutrients['Prot'] + chicken * chicken_nutrients['Prot'] + 
                    beans * beans_nutrients['Prot'] + milk * milk_nutrients['Prot']) / 100.0
    pack_fat[p] = (rice * rice_nutrients['Fat'] + chicken * chicken_nutrients['Fat'] + 
                   beans * beans_nutrients['Fat'] + milk * milk_nutrients['Fat']) / 100.0
    pack_vitC[p] = (rice * rice_nutrients['VitC'] + chicken * chicken_nutrients['VitC'] + 
                    beans * beans_nutrients['VitC'] + milk * milk_nutrients['VitC']) / 100.0
    pack_chicken[p] = chicken
    pack_beans[p] = beans
    pack_cost[p] = (rice * rice_nutrients['Cost'] + chicken * chicken_nutrients['Cost'] + 
                    beans * beans_nutrients['Cost'] + milk * milk_nutrients['Cost']) / 100.0

# Coefficients for 1g of vegetables
veg_cal_coeff = veg_nutrients['Cal'] / 100.0
veg_prot_coeff = veg_nutrients['Prot'] / 100.0
veg_vitC_coeff = veg_nutrients['VitC'] / 100.0
veg_cost_coeff = veg_nutrients['Cost'] / 100.0

# 3. Decision variables
x = model.addVars(packs, days, vtype=GRB.INTEGER, name="x")
v = model.addVars(days, vtype=GRB.CONTINUOUS, lb=100.0, name="v") # Vegetable grams per day
y = model.addVars(days, vtype=GRB.BINARY, name="y") # Incentive day indicator

# 4. Objective function: Minimize total weekly cost
total_cost = gp.quicksum(pack_cost[p] * x[p, j] for p in packs for j in days) + \
             gp.quicksum(veg_cost_coeff * v[j] for j in days)
model.setObjective(total_cost, GRB.MINIMIZE)

# 5. Constraints
# Weekly purchase minimums
model.addConstr(gp.quicksum(x['Pack A', j] for j in days) >= 3)
model.addConstr(gp.quicksum(x['Pack B', j] for j in days) >= 1)
model.addConstr(gp.quicksum(x['Pack C', j] for j in days) >= 1)
model.addConstr(gp.quicksum(x['Pack D', j] for j in days) >= 5)

# Exactly one incentive meal day
model.addConstr(gp.quicksum(y[j] for j in days) == 1)

for j in days:
    # Daily delivery limits
    model.addConstr(x['Pack A', j] <= 2)
    model.addConstr(x['Pack B', j] <= 5)
    model.addConstr(x['Pack C', j] <= 5)
    model.addConstr(x['Pack D', j] <= 1)
    
    # Nutritional constraints from packs + vegetables
    daily_cal = gp.quicksum(pack_cal[p] * x[p, j] for p in packs) + veg_cal_coeff * v[j]
    daily_prot = gp.quicksum(pack_prot[p] * x[p, j] for p in packs) + veg_prot_coeff * v[j]
    daily_fat = gp.quicksum(pack_fat[p] * x[p, j] for p in packs) # Vegetables have 0 fat
    daily_vitC = gp.quicksum(pack_vitC[p] * x[p, j] for p in packs) + veg_vitC_coeff * v[j]
    daily_chicken = gp.quicksum(pack_chicken[p] * x[p, j] for p in packs)
    daily_beans = gp.quicksum(pack_beans[p] * x[p, j] for p in packs)
    
    # Fixed nutritional requirements
    model.addConstr(daily_prot >= 50, name=f"prot_min_day_{j}")
    model.addConstr(daily_vitC >= 100, name=f"vitC_min_day_{j}")
    model.addConstr(daily_fat <= 70, name=f"fat_max_day_{j}")
    model.addConstr(daily_chicken <= 300, name=f"chicken_max_day_{j}")
    model.addConstr(daily_beans <= 400, name=f"beans_max_day_{j}")
    
    # Calorie bounds with indicators for Incentive (y[j]=1) vs Regular (y[j]=0)
    # Incentive day: [2500, 3000]
    model.addGenConstrIndicator(y[j], 1, daily_cal >= 2500)
    model.addGenConstrIndicator(y[j], 1, daily_cal <= 3000)
    # Regular day: [2000, 2500]
    model.addGenConstrIndicator(y[j], 0, daily_cal >= 2000)
    model.addGenConstrIndicator(y[j], 0, daily_cal <= 2500)

# 6. Solve and output the result
model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"Optimal objective value (Total Weekly Cost): {model.objVal}")
    print(f"FinalAnswer=【{model.objVal}】")