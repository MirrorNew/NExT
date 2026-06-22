import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
# Time periods
T_list = ['January', 'February', 'March', 'April', 'May', 'June']
num_months = len(T_list) # 6

# Products
J = 7 # Products I to VII, indexed 0 to 6

# Equipment quantities (Total)
total_machines = {
    'Grinding machine': 4,
    'Vertical drill': 2,
    'Horizontal drill': 3,
    'Boring machine': 1,
    'Planer': 1
}

# Repair Schedule
# Maps month name to a dictionary of {machine_key: count_repaired}
repair_schedule = {
    'January': {'grinder': 1},
    'February': {'horizontal drill': 2},
    'March': {'boring machine': 1},
    'April': {'vertical drill': 1},
    'May': {'grinder': 1, 'vertical drill': 1},
    'June': {'planer': 1, 'horizontal drill': 1}
}

# Mapping repair keys to standard machine names
repair_key_map = {
    'grinder': 'Grinding machine',
    'horizontal drill': 'Horizontal drill',
    'boring machine': 'Boring machine',
    'vertical drill': 'Vertical drill',
    'planer': 'Planer'
}

# Costs and Capacities
storage_fee_per_unit = 5.0
storage_capacity_per_product = 100
final_inventory_req = 50
hours_per_month = 24 * 2 * 8 # 384

# Unit Usage (Hours per unit)
# Replace None with 0.0
usage_data_raw = {
    'Grinding machine': [0.5, 0.7, None, None, 0.3, 0.2, 0.5],
    'Vertical drill': [0.1, 0.2, None, 0.3, None, 0.6, None],
    'Horizontal drill': [0.2, None, 0.8, None, None, None, 0.6],
    'Boring machine': [0.05, 0.03, None, 0.07, 0.1, None, 0.08],
    'Planer': [None, None, 0.01, None, 0.05, None, 0.05]
}

usage_data = {}
for m, row in usage_data_raw.items():
    usage_data[m] = [val if val is not None else 0.0 for val in row]

# Profits
profits = [100, 60, 80, 40, 110, 90, 30]

# Demand
demands = {
    'January': [500, 1000, 300, 300, 800, 200, 100],
    'February': [600, 500, 200, 0, 400, 300, 150],
    'March': [300, 600, 0, 0, 500, 400, 100],
    'April': [200, 300, 400, 500, 200, 0, 100],
    'May': [0, 100, 500, 100, 1000, 300, 0],
    'June': [500, 500, 100, 300, 1100, 500, 60]
}

# 2. Create Gurobi Model
model = gp.Model("Sunshine_Factory_Optimization")

# 3. Create Decision Variables
# x[t, j]: Production of product j in month t
x = model.addVars(num_months, J, vtype=GRB.CONTINUOUS, name="x")

# s[t, j]: Sales of product j in month t
s = model.addVars(num_months, J, vtype=GRB.CONTINUOUS, name="s")

# I[t, j]: Inventory of product j at the end of month t
I = model.addVars(num_months, J, vtype=GRB.CONTINUOUS, name="I")

# 4. Set up the Objective Function
# Maximize Total Profit = (Sales Revenue) - (Storage Costs for Jan-May)
# Note: No storage fee for end of June (index 5)
revenue = gp.quicksum(profits[j] * s[t, j] for t in range(num_months) for j in range(J))
storage_costs = gp.quicksum(storage_fee_per_unit * I[t, j] for t in range(num_months - 1) for j in range(J))

model.setObjective(revenue - storage_costs, GRB.MAXIMIZE)

# 5. Add Constraints

# Demand Constraints
for t in range(num_months):
    month_key = T_list[t]
    for j in range(J):
        model.addConstr(s[t, j] <= demands[month_key][j], name=f"Demand_{t}_{j}")

# Inventory Balance Constraints
# I_{t,j} = I_{t-1,j} + x_{t,j} - s_{t,j}
# For t=0 (Jan), I_{-1,j} = 0
for j in range(J):
    for t in range(num_months):
        prev_inventory = 0 if t == 0 else I[t-1, j]
        model.addConstr(I[t, j] == prev_inventory + x[t, j] - s[t, j], name=f"Balance_{t}_{j}")

# Inventory Capacity Constraints
for t in range(num_months):
    for j in range(J):
        model.addConstr(I[t, j] <= storage_capacity_per_product, name=f"StorageCap_{t}_{j}")

# Final Inventory Requirement (End of June, t=5)
for j in range(J):
    model.addConstr(I[5, j] == final_inventory_req, name=f"FinalInv_{j}")

# Machine Capacity Constraints
# Capacity = (Total Machines - Repaired Machines) * 384 hours
machine_names = list(total_machines.keys())

for t in range(num_months):
    month_name = T_list[t]
    
    # Get repair info for this month
    repairs_this_month = repair_schedule.get(month_name, {})
    
    for machine_name in machine_names:
        total_count = total_machines[machine_name]
        
        # Determine how many are being repaired
        repair_count = 0
        # Check against the keys in the repair schedule
        for r_key, count in repairs_this_month.items():
            # Standardize key using map if needed, or check direct match
            if repair_key_map.get(r_key) == machine_name:
                repair_count = count
                break
        
        available_machines = total_count - repair_count
        # Ensure non-negative availability (though data implies valid inputs)
        available_machines = max(0, available_machines)
        
        max_hours = available_machines * hours_per_month
        
        # Sum of usage: sum(unit_usage * production)
        usage_coeffs = usage_data[machine_name]
        lhs = gp.quicksum(usage_coeffs[j] * x[t, j] for j in range(J))
        
        model.addConstr(lhs <= max_hours, name=f"Cap_{machine_name}_{t}")

# 6. Solve the model and print results
model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"FinalAnswer=【{model.ObjVal}】")
else:
    print("Optimization was not successful.")