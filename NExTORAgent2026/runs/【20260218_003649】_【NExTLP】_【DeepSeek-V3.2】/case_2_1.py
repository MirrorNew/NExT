import gurobipy as gp

# 1. Define parameters from the provided list
T = ['January', 'February', 'March', 'April', 'May', 'June']
J = 7
num_grinders = 4
num_vertical_drills = 2
num_horizontal_drills = 3
num_boring_machines = 1
num_planers = 1

repair_schedule = {
    'January': {'grinder': 1},
    'February': {'horizontal drill': 2},
    'March': {'boring machine': 1},
    'April': {'vertical drill': 1},
    'May': {'grinder': 1, 'vertical drill': 1},
    'June': {'planer': 1, 'horizontal drill': 1}
}

storage_fee_per_unit = 5
storage_capacity_per_product = 100
initial_inventory = [0, 0, 0, 0, 0, 0, 0]
final_inventory_requirement = [50, 50, 50, 50, 50, 50, 50]
days_per_month = 24
shifts_per_day = 2
hours_per_shift = 8
storage_fee_end_of_June = 0

Table_1_C3 = {
    'Grinding machine': [0.5, 0.7, None, None, 0.3, 0.2, 0.5],
    'Vertical drill': [0.1, 0.2, None, 0.3, None, 0.6, None],
    'Horizontal drill': [0.2, None, 0.8, None, None, None, 0.6],
    'Boring machine': [0.05, 0.03, None, 0.07, 0.1, None, 0.08],
    'Planer': [None, None, 0.01, None, 0.05, None, 0.05],
    'Profit per piece': [100, 60, 80, 40, 110, 90, 30]
}

Table_2_C4 = {
    'January': [500, 1000, 300, 300, 800, 200, 100],
    'February': [600, 500, 200, 0, 400, 300, 150],
    'March': [300, 600, 0, 0, 500, 400, 100],
    'April': [200, 300, 400, 500, 200, 0, 100],
    'May': [0, 100, 500, 100, 1000, 300, 0],
    'June': [500, 500, 100, 300, 1100, 500, 60]
}

# Calculate monthly hours available
hours_per_month = days_per_month * shifts_per_day * hours_per_shift  # 384

# Create model
model = gp.Model("Sunshine_Machinery_Production_Planning")

# 2. Decision variables
x = {}  # production
s = {}  # sales
I = {}  # inventory

for t_idx, month in enumerate(T):
    for j in range(J):
        x[month, j] = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"x_{month}_{j+1}")
        s[month, j] = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"s_{month}_{j+1}")
        I[month, j] = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"I_{month}_{j+1}")

# 3. Objective function
profit_terms = []
storage_cost_terms = []

for t_idx, month in enumerate(T):
    for j in range(J):
        profit_terms.append(Table_1_C3['Profit per piece'][j] * s[month, j])
        # Storage fee only for months 1-5 (January to May)
        if t_idx < 5:  # January (index 0) to May (index 4)
            storage_cost_terms.append(storage_fee_per_unit * I[month, j])

model.setObjective(gp.quicksum(profit_terms) - gp.quicksum(storage_cost_terms), gp.GRB.MAXIMIZE)

# 4. Constraints
# 4.1 Capacity constraints
for t_idx, month in enumerate(T):
    # Calculate available machines for each equipment type
    # Grinders
    grinder_repair = repair_schedule[month].get('grinder', 0)
    available_grinders = num_grinders - grinder_repair
    
    # Vertical drills
    vert_repair = repair_schedule[month].get('vertical drill', 0)
    available_vert_drills = num_vertical_drills - vert_repair
    
    # Horizontal drills
    horz_repair = repair_schedule[month].get('horizontal drill', 0)
    available_horz_drills = num_horizontal_drills - horz_repair
    
    # Boring machines
    boring_repair = repair_schedule[month].get('boring machine', 0)
    available_boring = num_boring_machines - boring_repair
    
    # Planers
    planer_repair = repair_schedule[month].get('planer', 0)
    available_planers = num_planers - planer_repair
    
    # Grinding machine capacity
    grinder_hours = []
    for j in range(J):
        hours = Table_1_C3['Grinding machine'][j]
        if hours is not None:
            grinder_hours.append(hours * x[month, j])
    if grinder_hours:
        model.addConstr(gp.quicksum(grinder_hours) <= available_grinders * hours_per_month, 
                       name=f"Grinder_Capacity_{month}")
    
    # Vertical drill capacity
    vert_drill_hours = []
    for j in range(J):
        hours = Table_1_C3['Vertical drill'][j]
        if hours is not None:
            vert_drill_hours.append(hours * x[month, j])
    if vert_drill_hours:
        model.addConstr(gp.quicksum(vert_drill_hours) <= available_vert_drills * hours_per_month,
                       name=f"VertDrill_Capacity_{month}")
    
    # Horizontal drill capacity
    horz_drill_hours = []
    for j in range(J):
        hours = Table_1_C3['Horizontal drill'][j]
        if hours is not None:
            horz_drill_hours.append(hours * x[month, j])
    if horz_drill_hours:
        model.addConstr(gp.quicksum(horz_drill_hours) <= available_horz_drills * hours_per_month,
                       name=f"HorzDrill_Capacity_{month}")
    
    # Boring machine capacity
    boring_hours = []
    for j in range(J):
        hours = Table_1_C3['Boring machine'][j]
        if hours is not None:
            boring_hours.append(hours * x[month, j])
    if boring_hours:
        model.addConstr(gp.quicksum(boring_hours) <= available_boring * hours_per_month,
                       name=f"Boring_Capacity_{month}")
    
    # Planer capacity
    planer_hours = []
    for j in range(J):
        hours = Table_1_C3['Planer'][j]
        if hours is not None:
            planer_hours.append(hours * x[month, j])
    if planer_hours:
        model.addConstr(gp.quicksum(planer_hours) <= available_planers * hours_per_month,
                       name=f"Planer_Capacity_{month}")

# 4.2 Demand limit constraints
for t_idx, month in enumerate(T):
    for j in range(J):
        demand = Table_2_C4[month][j]
        model.addConstr(s[month, j] <= demand, name=f"Demand_{month}_{j+1}")

# 4.3 Inventory balance constraints
for t_idx, month in enumerate(T):
    for j in range(J):
        if t_idx == 0:  # January
            model.addConstr(I[month, j] == initial_inventory[j] + x[month, j] - s[month, j],
                           name=f"Inventory_Balance_{month}_{j+1}")
        else:
            prev_month = T[t_idx-1]
            model.addConstr(I[month, j] == I[prev_month, j] + x[month, j] - s[month, j],
                           name=f"Inventory_Balance_{month}_{j+1}")

# 4.4 Inventory capacity constraints
for t_idx, month in enumerate(T):
    for j in range(J):
        model.addConstr(I[month, j] <= storage_capacity_per_product,
                       name=f"Inventory_Capacity_{month}_{j+1}")

# 4.5 Final inventory requirement
last_month = T[-1]  # June
for j in range(J):
    model.addConstr(I[last_month, j] == final_inventory_requirement[j],
                   name=f"Final_Inventory_{j+1}")

# 5. Solve the model
model.optimize()

# 6. Print results
if model.status == gp.GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"Total Profit: {model.ObjVal:.2f} yuan")
    
    # Calculate total sales revenue and total storage cost for verification
    total_sales_revenue = 0
    total_storage_cost = 0
    
    for t_idx, month in enumerate(T):
        for j in range(J):
            total_sales_revenue += Table_1_C3['Profit per piece'][j] * s[month, j].X
            if t_idx < 5:  # January to May
                total_storage_cost += storage_fee_per_unit * I[month, j].X
    
    print(f"Total Sales Revenue: {total_sales_revenue:.2f} yuan")
    print(f"Total Storage Cost (Jan-May): {total_storage_cost:.2f} yuan")
    
    # Print production plan summary
    print("\nMonthly Production Summary:")
    for month in T:
        total_prod = sum(x[month, j].X for j in range(J))
        total_sales = sum(s[month, j].X for j in range(J))
        print(f"{month}: Production={total_prod:.0f}, Sales={total_sales:.0f}")
    
    # Print final inventory
    print("\nFinal Inventory (June):")
    for j in range(J):
        print(f"Product {j+1}: {I[last_month, j].X:.0f} units")
    
    # Output the answer as required
    print(f"FinalAnswer=【{model.ObjVal:.2f}】")
else:
    print("No optimal solution found.")
    print(f"FinalAnswer=【0】")