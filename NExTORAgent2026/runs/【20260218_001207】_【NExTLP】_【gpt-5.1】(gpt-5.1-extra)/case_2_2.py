import gurobipy as gp

# Sunshine Machinery Factory production planning with Gurobi
# The code uses only the given Parameters List and implements the validated model.

def main():
    # -----------------------------
    # 1. Parameters from Parameters List
    # -----------------------------
    num_grinders = 4
    num_vertical_drills = 2
    num_horizontal_drills = 3
    num_boring_machines = 1
    num_planers = 1

    products = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
    machines = ['Grinding machine', 'Vertical drill', 'Horizontal drill', 'Boring machine', 'Planer']

    # Unit hours per product per machine (index 0..6 for I..VII)
    unit_hours_grinding = [0.5, 0.7, 0.0, 0.0, 0.3, 0.2, 0.5]
    unit_hours_vertical_drill = [0.1, 0.2, 0.0, 0.3, 0.0, 0.6, 0.0]
    unit_hours_horizontal_drill = [0.2, 0.0, 0.8, 0.0, 0.0, 0.0, 0.6]
    unit_hours_boring_machine = [0.05, 0.03, 0.0, 0.07, 0.1, 0.0, 0.08]
    unit_hours_planer = [0.0, 0.0, 0.01, 0.0, 0.05, 0.0, 0.05]

    profit_per_piece = [100, 60, 80, 40, 110, 90, 30]

    maintenance_January = ['Grinding machine']
    maintenance_February = ['Horizontal drill', 'Horizontal drill']
    maintenance_March = ['Boring machine']
    maintenance_April = ['Vertical drill']
    maintenance_May = ['Grinding machine', 'Vertical drill']
    maintenance_June = ['Planer', 'Horizontal drill']

    Table_2_MaxMarketDemand = [
        {'Month': 'January',   'Demand': [500, 1000, 300, 300, 800, 200, 100]},
        {'Month': 'January_2', 'Demand': [600, 500, 200, 0, 400, 300, 150]},  # February
        {'Month': 'March',     'Demand': [300, 600, 0, 0, 500, 400, 100]},
        {'Month': 'April',     'Demand': [200, 300, 400, 500, 200, 0, 100]},
        {'Month': 'May',       'Demand': [0, 100, 500, 100, 1000, 300, 0]},
        {'Month': 'June',      'Demand': [500, 500, 100, 300, 1100, 500, 60]},
    ]

    monthly_storage_fee_per_piece = 5.0
    max_inventory_per_product = 100
    initial_inventory_each_product_January = 0
    required_final_inventory_each_product_end_of_June = 50
    working_days_per_month = 24
    shifts_per_day = 2
    hours_per_shift = 8
    hours_per_machine_per_month = 384
    storage_fee_end_of_June_required = 0  # no fee charged in June

    # -----------------------------
    # 2. Derived data
    # -----------------------------
    # Months indices 0..5 for Jan..Jun
    months = list(range(6))
    num_months = len(months)
    num_products = len(products)

    # Market demand matrix demand[i][t]
    demand = [[0.0 for _ in months] for _ in range(num_products)]
    for t_idx, month_record in enumerate(Table_2_MaxMarketDemand):
        demands_for_month = month_record['Demand']
        for i in range(num_products):
            demand[i][t_idx] = float(demands_for_month[i])

    # Machine availability each month (start with full counts)
    grinder_available = [num_grinders] * num_months
    vertical_available = [num_vertical_drills] * num_months
    horizontal_available = [num_horizontal_drills] * num_months
    boring_available = [num_boring_machines] * num_months
    planer_available = [num_planers] * num_months

    # Apply maintenance by month index (0: Jan, ..., 5: Jun)
    # January (index 0)
    for m in maintenance_January:
        if m == 'Grinding machine':
            grinder_available[0] -= 1
        elif m == 'Vertical drill':
            vertical_available[0] -= 1
        elif m == 'Horizontal drill':
            horizontal_available[0] -= 1
        elif m == 'Boring machine':
            boring_available[0] -= 1
        elif m == 'Planer':
            planer_available[0] -= 1

    # February (index 1)
    for m in maintenance_February:
        if m == 'Grinding machine':
            grinder_available[1] -= 1
        elif m == 'Vertical drill':
            vertical_available[1] -= 1
        elif m == 'Horizontal drill':
            horizontal_available[1] -= 1
        elif m == 'Boring machine':
            boring_available[1] -= 1
        elif m == 'Planer':
            planer_available[1] -= 1

    # March (index 2)
    for m in maintenance_March:
        if m == 'Grinding machine':
            grinder_available[2] -= 1
        elif m == 'Vertical drill':
            vertical_available[2] -= 1
        elif m == 'Horizontal drill':
            horizontal_available[2] -= 1
        elif m == 'Boring machine':
            boring_available[2] -= 1
        elif m == 'Planer':
            planer_available[2] -= 1

    # April (index 3)
    for m in maintenance_April:
        if m == 'Grinding machine':
            grinder_available[3] -= 1
        elif m == 'Vertical drill':
            vertical_available[3] -= 1
        elif m == 'Horizontal drill':
            horizontal_available[3] -= 1
        elif m == 'Boring machine':
            boring_available[3] -= 1
        elif m == 'Planer':
            planer_available[3] -= 1

    # May (index 4)
    for m in maintenance_May:
        if m == 'Grinding machine':
            grinder_available[4] -= 1
        elif m == 'Vertical drill':
            vertical_available[4] -= 1
        elif m == 'Horizontal drill':
            horizontal_available[4] -= 1
        elif m == 'Boring machine':
            boring_available[4] -= 1
        elif m == 'Planer':
            planer_available[4] -= 1

    # June (index 5)
    for m in maintenance_June:
        if m == 'Grinding machine':
            grinder_available[5] -= 1
        elif m == 'Vertical drill':
            vertical_available[5] -= 1
        elif m == 'Horizontal drill':
            horizontal_available[5] -= 1
        elif m == 'Boring machine':
            boring_available[5] -= 1
        elif m == 'Planer':
            planer_available[5] -= 1

    # Avoid negative availability
    grinder_available = [max(0, g) for g in grinder_available]
    vertical_available = [max(0, v) for v in vertical_available]
    horizontal_available = [max(0, h) for h in horizontal_available]
    boring_available = [max(0, b) for b in boring_available]
    planer_available = [max(0, p) for p in planer_available]

    # Monthly capacity (hours) for each machine type
    Cap_G = [g * hours_per_machine_per_month for g in grinder_available]
    Cap_VD = [v * hours_per_machine_per_month for v in vertical_available]
    Cap_HD = [h * hours_per_machine_per_month for h in horizontal_available]
    Cap_BM = [b * hours_per_machine_per_month for b in boring_available]
    Cap_PL = [p * hours_per_machine_per_month for p in planer_available]

    # -----------------------------
    # 3. Create model
    # -----------------------------
    model = gp.Model("Sunshine_Machinery_Production_Planning")

    # -----------------------------
    # 4. Decision variables
    # -----------------------------
    # x[i,t] = production of product i in month t
    # y[i,t] = sales of product i in month t
    # s[i,t] = end-of-month inventory of product i at month t
    x = model.addVars(num_products, num_months, lb=0.0, name="x")
    y = model.addVars(num_products, num_months, lb=0.0, name="y")
    s = model.addVars(num_products, num_months, lb=0.0, name="s")

    # -----------------------------
    # 5. Constraints
    # -----------------------------

    # Demand upper bounds: 0 <= y[i,t] <= demand[i][t]
    for i in range(num_products):
        for t in months:
            model.addConstr(y[i, t] <= demand[i][t],
                            name=f"demand_{i}_{t}")

    # Inventory balance
    # January (t=0): s[i,0] = initial + x[i,0] - y[i,0]
    for i in range(num_products):
        model.addConstr(
            s[i, 0] == initial_inventory_each_product_January + x[i, 0] - y[i, 0],
            name=f"inv_bal_{i}_0"
        )
        # Months 2..6 (t=1..5): s[i,t] = s[i,t-1] + x[i,t] - y[i,t]
        for t in range(1, num_months):
            model.addConstr(
                s[i, t] == s[i, t - 1] + x[i, t] - y[i, t],
                name=f"inv_bal_{i}_{t}"
            )

    # Inventory capacity and final inventory requirement
    for i in range(num_products):
        for t in months:
            model.addConstr(s[i, t] <= max_inventory_per_product,
                            name=f"inv_cap_{i}_{t}")
        # Final inventory at end of June (t=5) must be 50
        model.addConstr(
            s[i, num_months - 1] == required_final_inventory_each_product_end_of_June,
            name=f"final_inventory_{i}"
        )

    # Machine capacity constraints for each month t
    for t in months:
        # Grinder
        model.addConstr(
            gp.quicksum(unit_hours_grinding[i] * x[i, t] for i in range(num_products))
            <= Cap_G[t],
            name=f"cap_grinder_{t}"
        )

        # Vertical drill
        model.addConstr(
            gp.quicksum(unit_hours_vertical_drill[i] * x[i, t] for i in range(num_products))
            <= Cap_VD[t],
            name=f"cap_vertical_{t}"
        )

        # Horizontal drill
        model.addConstr(
            gp.quicksum(unit_hours_horizontal_drill[i] * x[i, t] for i in range(num_products))
            <= Cap_HD[t],
            name=f"cap_horizontal_{t}"
        )

        # Boring machine
        model.addConstr(
            gp.quicksum(unit_hours_boring_machine[i] * x[i, t] for i in range(num_products))
            <= Cap_BM[t],
            name=f"cap_boring_{t}"
        )

        # Planer
        model.addConstr(
            gp.quicksum(unit_hours_planer[i] * x[i, t] for i in range(num_products))
            <= Cap_PL[t],
            name=f"cap_planer_{t}"
        )

    # -----------------------------
    # 6. Objective: maximize total profit
    # -----------------------------
    # Revenue from sales: sum_{t=0..5} sum_{i} profit[i] * y[i,t]
    revenue_expr = gp.quicksum(
        profit_per_piece[i] * y[i, t]
        for i in range(num_products)
        for t in months
    )

    # Holding cost: 5 * sum_{t=0..4} sum_{i} s[i,t], no cost for t=5 (June)
    holding_cost_expr = monthly_storage_fee_per_piece * gp.quicksum(
        s[i, t]
        for i in range(num_products)
        for t in range(num_months - 1)  # 0..4
    )

    objective = revenue_expr - holding_cost_expr
    model.setObjective(objective, gp.GRB.MAXIMIZE)

    # -----------------------------
    # 7. Solve model
    # -----------------------------
    model.optimize()

    # -----------------------------
    # 8. Output results
    # -----------------------------
    if model.status == gp.GRB.OPTIMAL:
        optimal_profit = model.objVal
        print(f"Optimal objective (maximum total profit): {optimal_profit:.2f}")

        # Detailed plan (optional)
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        for t in months:
            print(f"\n=== Month {month_names[t]} ===")
            for i in range(num_products):
                prod = x[i, t].X
                sale = y[i, t].X
                inv = s[i, t].X
                if prod > 1e-6 or sale > 1e-6 or inv > 1e-6:
                    print(
                        f"Product {products[i]}: "
                        f"produce={prod:.2f}, sell={sale:.2f}, inventory_end={inv:.2f}"
                    )

        # ATTENTION 1: FinalAnswer (only the objective value requested)
        print(f"FinalAnswer=【{optimal_profit:.2f}】")
    else:
        # If not optimal, still print status and a placeholder FinalAnswer
        print(f"Optimization ended with status {model.status}")
        print("FinalAnswer=【NaN】")


if __name__ == "__main__":
    main()