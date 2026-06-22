import gurobipy as gp
from gurobipy import GRB

# 1. Define all parameter matrices and data inputs.
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

storage_fee_per_unit = 5.0
storage_capacity_per_product = 100
initial_inventory = [0, 0, 0, 0, 0, 0, 0]
final_inventory_requirement = [50, 50, 50, 50, 50, 50, 50]
days_per_month = 24
shifts_per_day = 2
hours_per_shift = 8
H = days_per_month * shifts_per_day * hours_per_shift  # 384 hours per machine

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

# Data processing: replace None with 0 in Table_1_C3 for calculation
for machine in Table_1_C3:
    Table_1_C3[machine] = [0 if v is None else v for v in Table_1_C3[machine]]

# 2. Create the Gurobi model.
model = gp.Model("Sunshine_Machinery")

# 3. Create decision variables.
x = model.addVars(T, range(J), lb=0, name="x")
s = model.addVars(T, range(J), lb=0, name="s")
I = model.addVars(T, range(J), lb=0, ub=storage_capacity_per_product, name="I")

# 4. Set up the objective function.
profits = gp.quicksum(Table_1_C3['Profit per piece'][j] * s[t, j] for t in T for j in range(J))
# No storage fee for the inventory at the end of June.
storage_costs = gp.quicksum(storage_fee_per_unit * I[t, j] for t_idx, t in enumerate(T) if t_idx < 5 for j in range(J))
model.setObjective(profits - storage_costs, GRB.MAXIMIZE)

# 5. Add all constraints.
machine_map = {
    'Grinding machine': 'grinder',
    'Vertical drill': 'vertical drill',
    'Horizontal drill': 'horizontal drill',
    'Boring machine': 'boring machine',
    'Planer': 'planer'
}

machine_counts = {
    'Grinding machine': num_grinders,
    'Vertical drill': num_vertical_drills,
    'Horizontal drill': num_horizontal_drills,
    'Boring machine': num_boring_machines,
    'Planer': num_planers
}

for t_idx, t in enumerate(T):
    # Machine Capacity Constraints
    for m_name in machine_map:
        repairs = repair_schedule.get(t, {}).get(machine_map[m_name], 0)
        available_capacity = (machine_counts[m_name] - repairs) * H
        model.addConstr(gp.quicksum(Table_1_C3[m_name][j] * x[t, j] for j in range(J)) <= available_capacity, 
                        name=f"Capacity_{t}_{m_name}")
    
    for j in range(J):
        # Demand Limit Constraint
        model.addConstr(s[t, j] <= Table_2_C4[t][j], name=f"Demand_{t}_{j}")
        
        # Inventory Balance Constraint
        if t_idx == 0:
            model.addConstr(I[t, j] == initial_inventory[j] + x[t, j] - s[t, j], name=f"InventoryBalance_{t}_{j}")
        else:
            model.addConstr(I[t, j] == I[T[t_idx-1], j] + x[t, j] - s[t, j], name=f"InventoryBalance_{t}_{j}")

# Final Inventory Requirement Constraint
for j in range(J):
    model.addConstr(I[T[-1], j] == final_inventory_requirement[j], name=f"FinalInventoryRequirement_{j}")

# 6. Solve the model and print results.
model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"FinalAnswer=【{model.ObjVal}】")