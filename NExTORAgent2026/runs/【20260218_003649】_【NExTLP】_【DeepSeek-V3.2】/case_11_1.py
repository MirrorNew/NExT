import gurobipy as gp

# Define all parameter matrices and data inputs from the Parameters List
Factories = ['Shenzhen', 'Vietnam']
CandidateWarehouses = ['CityA', 'CityB', 'CityC']
FixedOperatingCost = {'CityA': 500, 'CityB': 400, 'CityC': 300}
CostReductionTarget = 0.15
CustomerRegions = ['Singapore', 'Malaysia', 'Philippines']
AllowedRoutes_FactoryWarehouse = [['Shenzhen', 'CityA'], ['Shenzhen', 'CityB'], ['Shenzhen', 'CityC'], ['Vietnam', 'CityB'], ['Vietnam', 'CityC']]
AllowedRoutes_WarehouseCustomer = [['CityA', 'Singapore'], ['CityA', 'Malaysia'], ['CityB', 'Singapore'], ['CityB', 'Malaysia'], ['CityB', 'Philippines'], ['CityC', 'Singapore'], ['CityC', 'Malaysia'], ['CityC', 'Philippines']]
PriorityWarehouse = ['CityB']
MinWarehousesOpen = 2
Table_1 = {'ShenzhenFactory': 1000, 'VietnamFactory': 800, 'Customer1': 500, 'Customer2': 700, 'Customer3': 500}
Table_2 = {'CityA': {'Customer1': 3, 'Customer2': 4, 'Customer3': None}, 'CityB': {'Customer1': None, 'Customer2': 3, 'Customer3': 3}, 'CityC': {'Customer1': 3, 'Customer2': 5, 'Customer3': 2}}
Table_3 = {'CityA': {'Shenzhen': 2, 'Vietnam': None}, 'CityB': {'Shenzhen': 4, 'Vietnam': 1}, 'CityC': {'Shenzhen': 3, 'Vietnam': 2}}

# Map customer indices to regions for easier reference
customer_map = {1: 'Singapore', 2: 'Malaysia', 3: 'Philippines'}

# Create a Gurobi model
model = gp.Model("SupplyChainNetworkOptimization")

# Create decision variables
# Binary variables for warehouse opening
y_A = model.addVar(vtype=gp.GRB.BINARY, name="y_A")
y_B = model.addVar(vtype=gp.GRB.BINARY, name="y_B")
y_C = model.addVar(vtype=gp.GRB.BINARY, name="y_C")

# Continuous variables for factory to warehouse flows
x_S_A = model.addVar(lb=0, ub=1000, vtype=gp.GRB.CONTINUOUS, name="x_S_A")
x_S_B = model.addVar(lb=0, ub=1000, vtype=gp.GRB.CONTINUOUS, name="x_S_B")
x_S_C = model.addVar(lb=0, ub=1000, vtype=gp.GRB.CONTINUOUS, name="x_S_C")
x_V_A = model.addVar(lb=0, ub=800, vtype=gp.GRB.CONTINUOUS, name="x_V_A")
x_V_B = model.addVar(lb=0, ub=800, vtype=gp.GRB.CONTINUOUS, name="x_V_B")
x_V_C = model.addVar(lb=0, ub=800, vtype=gp.GRB.CONTINUOUS, name="x_V_C")

# Continuous variables for warehouse to customer flows
x_A_1 = model.addVar(lb=0, ub=500, vtype=gp.GRB.CONTINUOUS, name="x_A_1")
x_A_2 = model.addVar(lb=0, ub=700, vtype=gp.GRB.CONTINUOUS, name="x_A_2")
x_A_3 = model.addVar(lb=0, ub=500, vtype=gp.GRB.CONTINUOUS, name="x_A_3")
x_B_1 = model.addVar(lb=0, ub=500, vtype=gp.GRB.CONTINUOUS, name="x_B_1")
x_B_2 = model.addVar(lb=0, ub=700, vtype=gp.GRB.CONTINUOUS, name="x_B_2")
x_B_3 = model.addVar(lb=0, ub=500, vtype=gp.GRB.CONTINUOUS, name="x_B_3")
x_C_1 = model.addVar(lb=0, ub=500, vtype=gp.GRB.CONTINUOUS, name="x_C_1")
x_C_2 = model.addVar(lb=0, ub=700, vtype=gp.GRB.CONTINUOUS, name="x_C_2")
x_C_3 = model.addVar(lb=0, ub=500, vtype=gp.GRB.CONTINUOUS, name="x_C_3")

# Update the model to integrate variables
model.update()

# Set up the objective function
# Transportation cost from factory to warehouse
factory_warehouse_cost = (
    Table_3['CityA']['Shenzhen'] * x_S_A +
    Table_3['CityB']['Shenzhen'] * x_S_B +
    Table_3['CityC']['Shenzhen'] * x_S_C +
    Table_3['CityB']['Vietnam'] * x_V_B +
    Table_3['CityC']['Vietnam'] * x_V_C
)

# Transportation cost from warehouse to customer
warehouse_customer_cost = (
    Table_2['CityA']['Customer1'] * x_A_1 +
    Table_2['CityA']['Customer2'] * x_A_2 +
    Table_2['CityB']['Customer2'] * x_B_2 +
    Table_2['CityB']['Customer3'] * x_B_3 +
    Table_2['CityC']['Customer1'] * x_C_1 +
    Table_2['CityC']['Customer2'] * x_C_2 +
    Table_2['CityC']['Customer3'] * x_C_3
)

# Fixed operating costs
fixed_cost = (
    FixedOperatingCost['CityA'] * y_A +
    FixedOperatingCost['CityB'] * y_B +
    FixedOperatingCost['CityC'] * y_C
)

# Total cost
total_cost = factory_warehouse_cost + warehouse_customer_cost + fixed_cost

# Set objective to minimize total cost
model.setObjective(total_cost, gp.GRB.MINIMIZE)

# Add all constraints
# 1. Demand satisfaction constraints
model.addConstr(x_A_1 + x_B_1 + x_C_1 == Table_1['Customer1'], "Demand_Customer1")
model.addConstr(x_A_2 + x_B_2 + x_C_2 == Table_1['Customer2'], "Demand_Customer2")
model.addConstr(x_A_3 + x_B_3 + x_C_3 == Table_1['Customer3'], "Demand_Customer3")

# 2. Factory capacity constraints
model.addConstr(x_S_A + x_S_B + x_S_C <= Table_1['ShenzhenFactory'], "Capacity_Shenzhen")
model.addConstr(x_V_A + x_V_B + x_V_C <= Table_1['VietnamFactory'], "Capacity_Vietnam")

# 3. Prohibited routes constraints
# Vietnam factory to warehouse A is not allowed
model.addConstr(x_V_A == 0, "No_Vietnam_to_A")
# Warehouse A to customer 3 is not allowed
model.addConstr(x_A_3 == 0, "No_A_to_Customer3")
# Warehouse B to customer 1 is not allowed
model.addConstr(x_B_1 == 0, "No_B_to_Customer1")

# 4. Flow balance constraints at each warehouse
model.addConstr(x_S_A + x_V_A == x_A_1 + x_A_2 + x_A_3, "FlowBalance_A")
model.addConstr(x_S_B + x_V_B == x_B_1 + x_B_2 + x_B_3, "FlowBalance_B")
model.addConstr(x_S_C + x_V_C == x_C_1 + x_C_2 + x_C_3, "FlowBalance_C")

# 5. Warehouse activation coupling constraints
# Use a large M value (max total flow possible)
M = 1800  # 1000 + 800

# Inbound coupling
model.addConstr(x_S_A <= M * y_A, "InboundCoupling_S_A")
model.addConstr(x_S_B <= M * y_B, "InboundCoupling_S_B")
model.addConstr(x_S_C <= M * y_C, "InboundCoupling_S_C")
model.addConstr(x_V_A <= M * y_A, "InboundCoupling_V_A")
model.addConstr(x_V_B <= M * y_B, "InboundCoupling_V_B")
model.addConstr(x_V_C <= M * y_C, "InboundCoupling_V_C")

# Outbound coupling
model.addConstr(x_A_1 <= M * y_A, "OutboundCoupling_A_1")
model.addConstr(x_A_2 <= M * y_A, "OutboundCoupling_A_2")
model.addConstr(x_A_3 <= M * y_A, "OutboundCoupling_A_3")
model.addConstr(x_B_1 <= M * y_B, "OutboundCoupling_B_1")
model.addConstr(x_B_2 <= M * y_B, "OutboundCoupling_B_2")
model.addConstr(x_B_3 <= M * y_B, "OutboundCoupling_B_3")
model.addConstr(x_C_1 <= M * y_C, "OutboundCoupling_C_1")
model.addConstr(x_C_2 <= M * y_C, "OutboundCoupling_C_2")
model.addConstr(x_C_3 <= M * y_C, "OutboundCoupling_C_3")

# 6. Mandatory opening of warehouse B
model.addConstr(y_B == 1, "Open_Warehouse_B")

# 7. Minimum warehouses opened constraint
model.addConstr(y_A + y_B + y_C >= MinWarehousesOpen, "Min_Warehouses_Open")

# Note: The cost reduction constraint (Z ≤ 0.85 * C0) is not added because C0 (baseline total cost before optimization) 
# is not provided in the parameters list. This constraint would require knowing the baseline cost.

# Solve the model
model.optimize()

# Print results
if model.status == gp.GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"Total cost: {model.ObjVal:.2f} yuan")
    
    print("\nWarehouse opening decisions:")
    print(f"Warehouse A (CityA) opened: {y_A.X}")
    print(f"Warehouse B (CityB) opened: {y_B.X}")
    print(f"Warehouse C (CityC) opened: {y_C.X}")
    
    print("\nFactory to warehouse flows:")
    print(f"Shenzhen to A: {x_S_A.X:.2f}")
    print(f"Shenzhen to B: {x_S_B.X:.2f}")
    print(f"Shenzhen to C: {x_S_C.X:.2f}")
    print(f"Vietnam to A: {x_V_A.X:.2f}")
    print(f"Vietnam to B: {x_V_B.X:.2f}")
    print(f"Vietnam to C: {x_V_C.X:.2f}")
    
    print("\nWarehouse to customer flows:")
    print(f"A to Customer1: {x_A_1.X:.2f}")
    print(f"A to Customer2: {x_A_2.X:.2f}")
    print(f"A to Customer3: {x_A_3.X:.2f}")
    print(f"B to Customer1: {x_B_1.X:.2f}")
    print(f"B to Customer2: {x_B_2.X:.2f}")
    print(f"B to Customer3: {x_B_3.X:.2f}")
    print(f"C to Customer1: {x_C_1.X:.2f}")
    print(f"C to Customer2: {x_C_2.X:.2f}")
    print(f"C to Customer3: {x_C_3.X:.2f}")
    
    # The question asks for the answer (minimized total cost)
    print(f"FinalAnswer=【{model.ObjVal:.2f}】")
else:
    print("No optimal solution found.")
    print(f"FinalAnswer=【No feasible solution】")