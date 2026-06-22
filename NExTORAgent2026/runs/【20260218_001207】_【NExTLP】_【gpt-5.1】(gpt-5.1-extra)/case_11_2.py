import gurobipy as gp

# Solve the given MILP supply chain network design problem with Gurobi
# and print the final optimal total cost as FinalAnswer.
if __name__ == "__main__":
    # ========== 1. Parameters (strictly use given Values) ==========
    num_factories = 2
    factories = ['Shenzhen Factory 1', 'Vietnam Factory']

    num_warehouses = 3
    warehouses = ['City A', 'City B', 'City C']

    num_customers = 3
    customers_by_name = ['Singapore', 'Malaysia', 'Philippines']

    fixed_cost_warehouse = [500, 400, 300]
    required_cost_reduction_percentage = 0.15   # given, but no baseline provided

    factory_capacity = [1000, 800]              # [Shenzhen, Vietnam]
    customer_demand = [500, 700, 500]           # [Customer1, Customer2, Customer3]

    # transport_cost_warehouse_to_customer[i][j]: warehouse i to customer j
    # This is exactly the given Parameters List value.
    transport_cost_warehouse_to_customer = [
        [3, 4, None],   # City A → [C1, C2, C3]
        [None, 3, 3],   # City B → [C1, C2, C3]
        [3, 5, 2]       # City C → [C1, C2, C3]
    ]

    # transport_cost_factory_to_warehouse table from Parameters List:
    # Value = [[2, None], [4, 1], [3, 2]]
    # We must not rewrite it or change numeric values, but we need a 2×3
    # (factory × warehouse) matrix to match the model.
    #
    # Based on the text:
    #   Shenzhen can ship to A,B,C. Vietnam can ship only to B,C.
    # So we map the 3-row list into a 2×3 matrix consistently while
    # preserving all provided numbers and Nones.
    #
    # Here we interpret:
    #   - first column of each row used for Shenzhen→A,B,C
    #   - second column of rows 2 and 3 used for Vietnam→B,C
    raw_fw = [[2, None], [4, 1], [3, 2]]
    transport_cost_factory_to_warehouse = [
        [raw_fw[0][0], raw_fw[1][0], raw_fw[2][0]],  # Shenzhen → A,B,C = [2,4,3]
        [None,         raw_fw[1][1], raw_fw[2][1]]   # Vietnam  → -,B,C = [None,1,2]
    ]

    # Allowed arcs from factory to warehouse (given)
    factory_to_warehouse_allowed = [
        [1, 1, 1],  # Shenzhen → A,B,C
        [0, 1, 1]   # Vietnam  →  -,B,C
    ]

    # Allowed arcs from warehouse to customer (given)
    warehouse_to_customer_allowed = [
        [1, 1, 0],  # City A → C1,C2
        [0, 1, 1],  # City B →   C2,C3
        [1, 1, 1]   # City C → C1,C2,C3
    ]

    min_warehouses_open = 2

    # A safe Big-M: at least max(total demand, total capacity)
    total_demand = sum(customer_demand)
    total_capacity = sum(factory_capacity)
    M = max(total_demand, total_capacity)

    # ========== 2. Create model ==========
    model = gp.Model("SupplyChain_Warehouse_Location_and_Routing")

    # ========== 3. Decision variables ==========

    # y[i]: 1 if warehouse i is opened
    y = model.addVars(
        range(num_warehouses),
        vtype=gp.GRB.BINARY,
        name="y"
    )

    # x[k,i]: quantity from factory k to warehouse i (only if allowed)
    x = {}
    for k in range(num_factories):
        for i in range(num_warehouses):
            if factory_to_warehouse_allowed[k][i] == 1:
                x[k, i] = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS,
                                       name=f"x_{k}_{i}")
            # forbidden arcs: no variable (implicitly 0)

    # z[i,j]: quantity from warehouse i to customer j (only if allowed)
    z = {}
    for i in range(num_warehouses):
        for j in range(num_customers):
            if warehouse_to_customer_allowed[i][j] == 1:
                z[i, j] = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS,
                                       name=f"z_{i}_{j}")
            # forbidden arcs: no variable

    model.update()

    # ========== 4. Objective: minimize total cost ==========

    obj = gp.LinExpr()

    # Fixed warehouse costs
    for i in range(num_warehouses):
        obj.addTerms(fixed_cost_warehouse[i], y[i])

    # Factory → Warehouse costs
    for k in range(num_factories):
        for i in range(num_warehouses):
            if (k, i) in x:
                cost_fw = transport_cost_factory_to_warehouse[k][i]
                # cost_fw may be None on some arcs; skip those
                if cost_fw is not None:
                    obj.addTerms(cost_fw, x[k, i])

    # Warehouse → Customer costs
    for i in range(num_warehouses):
        for j in range(num_customers):
            if (i, j) in z:
                cost_wc = transport_cost_warehouse_to_customer[i][j]
                if cost_wc is not None:
                    obj.addTerms(cost_wc, z[i, j])

    model.setObjective(obj, gp.GRB.MINIMIZE)

    # ========== 5. Constraints ==========

    # --- Factory capacity ---
    # Shenzhen (k=0)
    model.addConstr(
        gp.quicksum(x[k, i] for i in range(num_warehouses)
                    if (0, i) in x) <= factory_capacity[0],
        name="Shenzhen_capacity"
    )

    # Vietnam (k=1)
    model.addConstr(
        gp.quicksum(x[1, i] for i in range(num_warehouses)
                    if (1, i) in x) <= factory_capacity[1],
        name="Vietnam_capacity"
    )

    # --- Customer demand satisfaction ---
    # Customer 1 (Singapore, j=0)
    model.addConstr(
        gp.quicksum(z[i, 0] for i in range(num_warehouses)
                    if (i, 0) in z) == customer_demand[0],
        name="Demand_C1"
    )

    # Customer 2 (Malaysia, j=1)
    model.addConstr(
        gp.quicksum(z[i, 1] for i in range(num_warehouses)
                    if (i, 1) in z) == customer_demand[1],
        name="Demand_C2"
    )

    # Customer 3 (Philippines, j=2)
    model.addConstr(
        gp.quicksum(z[i, 2] for i in range(num_warehouses)
                    if (i, 2) in z) == customer_demand[2],
        name="Demand_C3"
    )

    # --- Warehouse flow conservation ---
    # inbound (sum from factories) = outbound (sum to customers) for each warehouse
    for i in range(num_warehouses):
        inbound = gp.quicksum(x[k, i] for k in range(num_factories)
                              if (k, i) in x)
        outbound = gp.quicksum(z[i, j] for j in range(num_customers)
                               if (i, j) in z)
        model.addConstr(inbound == outbound, name=f"Flow_warehouse_{i}")

    # --- Activation linking (Big-M) ---
    for i in range(num_warehouses):
        # inbound arcs
        for k in range(num_factories):
            if (k, i) in x:
                model.addConstr(
                    x[k, i] <= M * y[i],
                    name=f"Link_in_{k}_{i}"
                )
        # outbound arcs
        for j in range(num_customers):
            if (i, j) in z:
                model.addConstr(
                    z[i, j] <= M * y[i],
                    name=f"Link_out_{i}_{j}"
                )

    # --- Policy constraints ---
    # Warehouse B (index 1) must be opened
    model.addConstr(y[1] == 1, name="Warehouse_B_must_open")

    # At least 2 warehouses open
    model.addConstr(
        gp.quicksum(y[i] for i in range(num_warehouses)) >= min_warehouses_open,
        name="Min_warehouses_open"
    )

    # NOTE: required_cost_reduction_percentage is not enforced explicitly
    # because no baseline cost is provided in the data.

    # ========== 6. Optimize ==========
    model.optimize()

    # ========== 7. Print results and FinalAnswer ==========
    if model.status == gp.GRB.OPTIMAL:
        # Main answer: optimal total cost
        optimal_total_cost = model.objVal

        # (Optional) detailed outputs
        print(f"Optimal objective value (total cost): {optimal_total_cost:.2f}")
        print("\nWarehouse open decisions:")
        for i in range(num_warehouses):
            print(f"  {warehouses[i]}: y[{i}] = {int(round(y[i].X))}")

        print("\nFactory → Warehouse shipments:")
        for (k, i), var in x.items():
            if var.X > 1e-6:
                print(f"  {factories[k]} → {warehouses[i]}: {var.X:.2f}")

        print("\nWarehouse → Customer shipments:")
        for (i, j), var in z.items():
            if var.X > 1e-6:
                print(f"  {warehouses[i]} → {customers_by_name[j]}: {var.X:.2f}")

        # Required final output format:
        print(f"FinalAnswer=【{optimal_total_cost:.2f}】")
    else:
        print(f"Optimization ended with status {model.status}")
        # If infeasible/non-optimal, still output something for FinalAnswer
        print("FinalAnswer=【NaN】")