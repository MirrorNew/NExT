import gurobipy as gp
from gurobipy import GRB


def solve_timber_storage():
    """
    Solves the timber storage and transport problem to maximize annual profit
    with a non-linear holding cost: (70 + 100 * u^1.2) yuan/m³,
    where u is storage time in quarters.
    """
    try:
        # --- Data ---
        # Quarters: Winter (0), Spring (1), Summer (2), Autumn (3)
        quarters = list(range(4))
        quarter_names = ['Winter', 'Spring', 'Summer', 'Autumn']

        # Prices and Sales Limits (Units: 10k yuan for price, 10k m^3 for volume)
        purchase_prices = [410, 430, 460, 450]  # P_t
        sale_prices = [425, 440, 465, 455]      # S_t
        max_sales_volume = [100, 140, 200, 160]  # MaxSales_t

        # Warehouse Capacity (Units: 10k m^3) — 200000 m³ = 20 * 10k m³
        warehouse_capacity = 20

        # Initial Inventory (Units: 10k m^3)
        initial_inventory = 0.0

        # --- Precompute per-quarter holding cost coefficients (10k yuan / (10k m³·quarter)) ---
        # Original cost form (yuan/m³): 70 + 100 * u^1.2
        # Convert to 10k yuan / 10k m³ by dividing by 10000:
        # H_u = (70 + 100 * u^1.2) / 10000  (10k yuan / 10k m³ for storage duration u)
        #
        # For each quarter t, inventory Inv[t] is held for 1 quarter, but items in Inv[t]
        # may have been stored for multiple quarters (t - purchase_quarter + 1).
        #
        # To keep the model reasonably simple and still reflect non-linearity over time,
        # we approximate by charging inventory in quarter t with a unit cost that depends
        # on how long it has been in storage:
        #   u_t = t + 1  (Winter inventory: u=1; Spring: u=2; Summer: u=3; Autumn: u=4)
        #   H_t = (70 + 100 * u_t^1.2) / 10000
        #
        # This keeps the model linear in the decision variables but non-linear in time,
        # which is the part specified by the new problem statement.
        u_list = [t + 1 for t in quarters]  # Storage age proxy for Inv[t]
        storage_cost_per_unit_volume = [
            (70.0 + 100.0 * (u )) / 10000.0 for u in u_list
        ]  # length = 4

        # --- Create Gurobi Model ---
        model = gp.Model("TimberStorageOptimization_NonlinearCostInTime")

        # --- Decision Variables ---
        # Buy[t]: Volume purchased at the beginning of quarter t
        buy_vars = model.addVars(quarters,
                                 name="Buy",
                                 lb=0.0,
                                 vtype=GRB.CONTINUOUS)

        # Sell[t]: Volume sold during quarter t
        sell_vars = model.addVars(quarters,
                                  name="Sell",
                                  lb=0.0,
                                  vtype=GRB.CONTINUOUS)

        # Inv[t]: Volume stored at the end of quarter t
        inventory_vars = model.addVars(quarters,
                                       name="Inventory",
                                       lb=0.0,
                                       vtype=GRB.CONTINUOUS)

        # --- Objective Function: Maximize Total Profit ---
        # Profit = Sales Revenue - Purchase Cost - Holding Cost
        total_sales_revenue = gp.quicksum(sale_prices[t] * sell_vars[t]
                                          for t in quarters)
        total_purchase_cost = gp.quicksum(purchase_prices[t] * buy_vars[t]
                                          for t in quarters)

        # ❤ Non-linearity is introduced. ❤
        # total_holding_cost = gp.quicksum(
        #     storage_cost_per_unit_volume * inventory_vars[t] for t in quarters)
        # New holding cost: per-quarter coefficient depends on effective storage time u_t,
        # following (70 + 100 * u_t^1.2)/10000 in 10k yuan per 10k m³.
        total_holding_cost = gp.quicksum(
            storage_cost_per_unit_volume[t] * inventory_vars[t]
            for t in quarters
        )

        model.setObjective(
            total_sales_revenue - total_purchase_cost - total_holding_cost,
            GRB.MAXIMIZE)

        # --- Constraints ---
        for t in quarters:
            # Inventory at the start of the current quarter t
            prev_inventory = initial_inventory if t == 0 else inventory_vars[t - 1]

            # 1. Inventory Balance Constraint
            # Inv[t] = Inv[t-1] + Buy[t] - Sell[t]
            model.addConstr(
                inventory_vars[t] == prev_inventory + buy_vars[t] - sell_vars[t],
                name=f"InventoryBalance_{quarter_names[t]}"
            )

            # 2. Warehouse Capacity Constraint
            # Inv[t] <= warehouse_capacity
            model.addConstr(
                inventory_vars[t] <= warehouse_capacity,
                name=f"WarehouseCapacity_{quarter_names[t]}"
            )

            # 3. Sales Limit Constraint
            # Sell[t] <= MaxSales[t]
            model.addConstr(
                sell_vars[t] <= max_sales_volume[t],
                name=f"MaxSales_{quarter_names[t]}"
            )

            # 4. Availability for Sale Constraint
            # Sell[t] <= Inv[t-1] + Buy[t]
            model.addConstr(
                sell_vars[t] <= prev_inventory + buy_vars[t],
                name=f"SalesAvailability_{quarter_names[t]}"
            )

        # 5. End Condition: All inventory must be sold by the end of Autumn (t=3)
        model.addConstr(inventory_vars[quarters[-1]] == 0,
                        name="EndInventoryZero")

        # Suppress Gurobi output to console if desired
        # model.setParam('OutputFlag', 0)

        # Optimize the model
        model.optimize()

        # --- Results ---
        if model.status == GRB.OPTIMAL:
            print("Optimal storage and transport plan found.")
            print(f"Maximum Annual Profit: {model.ObjVal * 10000:.2f} Yuan")

            print("\nQuarterly Plan Details (Volumes in 10,000 m³):")
            header = (
                f"{'Quarter':<8} | {'Inv Start':<10} | {'Buy Qty':<10} | "
                f"{'Sell Qty':<10} | {'Inv End':<10} | {'Unit H Cost':<12} | {'Holding Cost':<15}"
            )
            print(header)
            print("-" * len(header))

            current_inv = initial_inventory
            for t in quarters:
                unit_h_cost_t = storage_cost_per_unit_volume[t]  # 10k yuan / 10k m³
                holding_cost_t_10k = unit_h_cost_t * inventory_vars[t].X
                print(
                    f"{quarter_names[t]:<8} | {current_inv:<10.2f} | {buy_vars[t].X:<10.2f} | "
                    f"{sell_vars[t].X:<10.2f} | {inventory_vars[t].X:<10.2f} | "
                    f"{unit_h_cost_t:<12.4f} | {holding_cost_t_10k * 10000:<15.2f}"
                )
                current_inv = inventory_vars[t].X
            print("-" * len(header))

            print(
                f"\nInitial Inventory (Start of Winter): {initial_inventory:.2f} (10k m³)"
            )
            print(
                f"Final Inventory (End of Autumn): {inventory_vars[quarters[-1]].X:.2f} (10k m³)"
            )

            # Cost breakdown
            sales_rev_val = sum(sale_prices[t] * sell_vars[t].X
                                for t in quarters) * 10000
            purchase_cost_val = sum(purchase_prices[t] * buy_vars[t].X
                                    for t in quarters) * 10000
            holding_cost_val = sum(
                storage_cost_per_unit_volume[t] * inventory_vars[t].X
                for t in quarters) * 10000
            print("\nFinancial Summary (Yuan):")
            print(f"  Total Sales Revenue: {sales_rev_val:.2f}")
            print(f"  Total Purchase Cost: {purchase_cost_val:.2f}")
            print(f"  Total Holding Cost: {holding_cost_val:.2f}")
            print(
                f"  Calculated Profit: {(sales_rev_val - purchase_cost_val - holding_cost_val):.2f}"
            )

        elif model.status == GRB.INFEASIBLE:
            print(
                "Model is infeasible. Check constraints and data for contradictions."
            )
        else:
            print(f"Optimization stopped with status: {model.status}")

    except gp.GurobiError as e:
        print(f"Gurobi error code {e.errno}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    solve_timber_storage()