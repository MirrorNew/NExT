import gurobipy as gp
from gurobipy import GRB

# 1. Define Data and Parameters
# Nutrient and Cost data per 100g
# Format: [Calories, Protein, Fat, Vitamin C, Cost]
raw_ingredients = {
    'Rice':       {'Calories': 360, 'Protein': 6,  'Fat': 1,  'Vitamin C': 0,  'Cost': 0.50},
    'Chicken':    {'Calories': 200, 'Protein': 20, 'Fat': 15, 'Vitamin C': 0,  'Cost': 2.00},
    'Beans':      {'Calories': 150, 'Protein': 8,  'Fat': 1,  'Vitamin C': 5,  'Cost': 1.00},
    'Milk':       {'Calories': 60,  'Protein': 3,  'Fat': 4,  'Vitamin C': 5,  'Cost': 1.50},
    'Vegetables': {'Calories': 50,  'Protein': 2,  'Fat': 0,  'Vitamin C': 20, 'Cost': 1.00}
}

# Pack compositions in grams
pack_composition = {
    'Pack A': {'Rice': 25, 'Chicken': 50, 'Beans': 0,  'Milk': 0},
    'Pack B': {'Rice': 25, 'Chicken': 40, 'Beans': 0,  'Milk': 0},
    'Pack C': {'Rice': 10, 'Chicken': 20, 'Beans': 20, 'Milk': 0},
    'Pack D': {'Rice': 0,  'Chicken': 0,  'Beans': 0,  'Milk': 50}
}

# Daily Limits (Upper Bound for usage) and Weekly Limits (Lower Bound for purchase)
pack_limits = {
    'Pack A': {'DailyMax': 2, 'WeeklyMin': 3},
    'Pack B': {'DailyMax': 5, 'WeeklyMin': 1},
    'Pack C': {'DailyMax': 5, 'WeeklyMin': 1},
    'Pack D': {'DailyMax': 1, 'WeeklyMin': 5}
}

# Helper function to calculate nutrient content for a pack
def get_pack_attributes(pack_name):
    comp = pack_composition[pack_name]
    attrs = {'Calories': 0, 'Protein': 0, 'Fat': 0, 'Vitamin C': 0, 'Cost': 0, 
             'Chicken_g': 0, 'Beans_g': 0}
    
    for ing, grams in comp.items():
        if grams > 0:
            factor = grams / 100.0
            attrs['Calories']  += raw_ingredients[ing]['Calories'] * factor
            attrs['Protein']   += raw_ingredients[ing]['Protein'] * factor
            attrs['Fat']       += raw_ingredients[ing]['Fat'] * factor
            attrs['Vitamin C'] += raw_ingredients[ing]['Vitamin C'] * factor
            attrs['Cost']      += raw_ingredients[ing]['Cost'] * factor
            
            if ing == 'Chicken':
                attrs['Chicken_g'] += grams
            if ing == 'Beans':
                attrs['Beans_g'] += grams
    return attrs

# Helper function for Vegetables per gram (since variable v_j is in grams)
def get_veg_attributes_per_gram():
    factor = 1.0 / 100.0
    ing = raw_ingredients['Vegetables']
    return {
        'Calories':  ing['Calories'] * factor,
        'Protein':   ing['Protein'] * factor,
        'Fat':       ing['Fat'] * factor,
        'Vitamin C': ing['Vitamin C'] * factor,
        'Cost':      ing['Cost'] * factor
    }

# Pre-calculate attributes
pack_attrs = {p: get_pack_attributes(p) for p in pack_composition}
veg_attrs = get_veg_attributes_per_gram()

# 2. Create Model
model = gp.Model("Healthy_Meal_Optimization")

# 3. Decision Variables
days = range(1, 8) # Days 1 to 7
packs = ['Pack A', 'Pack B', 'Pack C', 'Pack D']

# x[p, j]: Number of packs of type p on day j
x = model.addVars(packs, days, vtype=GRB.INTEGER, lb=0, name="x")

# v[j]: Grams of vegetables on day j
v = model.addVars(days, vtype=GRB.CONTINUOUS, lb=100, name="v") # Min 100g/day handled by lb

# y[j]: Indicator for incentive meal on day j
y = model.addVars(days, vtype=GRB.BINARY, name="y")

# 4. Set Objective: Minimize Total Weekly Cost
total_cost = gp.quicksum(
    pack_attrs[p]['Cost'] * x[p, j] for p in packs for j in days
) + gp.quicksum(
    veg_attrs['Cost'] * v[j] for j in days
)
model.setObjective(total_cost, GRB.MINIMIZE)

# 5. Constraints

# 5.1 Incentive Meal Constraints
# Exactly one incentive meal day per week
model.addConstr(gp.quicksum(y[j] for j in days) == 1, "One_Incentive_Day")

# 5.2 Daily Constraints
for j in days:
    # Expressions for daily totals
    daily_cal = gp.quicksum(pack_attrs[p]['Calories'] * x[p, j] for p in packs) + veg_attrs['Calories'] * v[j]
    daily_prot = gp.quicksum(pack_attrs[p]['Protein'] * x[p, j] for p in packs) + veg_attrs['Protein'] * v[j]
    daily_fat = gp.quicksum(pack_attrs[p]['Fat'] * x[p, j] for p in packs) + veg_attrs['Fat'] * v[j]
    daily_vitc = gp.quicksum(pack_attrs[p]['Vitamin C'] * x[p, j] for p in packs) + veg_attrs['Vitamin C'] * v[j]
    
    daily_chicken = gp.quicksum(pack_attrs[p]['Chicken_g'] * x[p, j] for p in packs)
    daily_beans = gp.quicksum(pack_attrs[p]['Beans_g'] * x[p, j] for p in packs)
    
    # Calorie Limits with Incentive Shift
    # Base: [2000, 2500]. Incentive: [2500, 3000] (Base + 500)
    # Lower Bound
    model.addConstr(daily_cal >= 2000 + 500 * y[j], f"Cal_Min_Day_{j}")
    # Upper Bound
    model.addConstr(daily_cal <= 2500 + 500 * y[j], f"Cal_Max_Day_{j}")
    
    # Nutrient Requirements
    model.addConstr(daily_prot >= 50, f"Prot_Min_Day_{j}")
    model.addConstr(daily_vitc >= 100, f"VitC_Min_Day_{j}")
    model.addConstr(daily_fat <= 70, f"Fat_Max_Day_{j}")
    
    # Ingredient Limits
    # Vegetables >= 100 is handled by variable lower bound (lb=100)
    model.addConstr(daily_chicken <= 300, f"Chicken_Max_Day_{j}")
    model.addConstr(daily_beans <= 400, f"Beans_Max_Day_{j}")
    
    # Pack Daily Usage Limits
    for p in packs:
        model.addConstr(x[p, j] <= pack_limits[p]['DailyMax'], f"Daily_Limit_{p}_{j}")

# 5.3 Weekly Purchase Constraints
for p in packs:
    model.addConstr(gp.quicksum(x[p, j] for j in days) >= pack_limits[p]['WeeklyMin'], f"Weekly_Min_{p}")

# 6. Solve
model.optimize()

# 7. Output
if model.status == GRB.OPTIMAL:
    # print("Optimal Solution Found:")
    # for j in days:
    #     day_type = "Incentive" if y[j].X > 0.5 else "Standard"
    #     print(f"Day {j} ({day_type}):")
    #     for p in packs:
    #         if x[p, j].X > 0:
    #             print(f"  {p}: {x[p, j].X}")
    #     print(f"  Vegetables: {v[j].X:.2f} g")
    #     print(f"  Cost: {sum(pack_attrs[p]['Cost']*x[p,j].X for p in packs) + veg_attrs['Cost']*v[j].X:.2f}")
    
    print(f"FinalAnswer=【{model.objVal}】")
else:
    print("No optimal solution found.")