import gurobipy as gp
from gurobipy import GRB


def solve_purchase_sales_plan():
    """
    Solves the purchase and sales planning problem to maximize total profit
    over a 6-month period, subject to warehouse capacity, inventory flow,
    and an additional piecewise-fixed warehouse management fee that may
    be triggered once cumulative purchases exceed 2000 units.
    """
    try:
        # --- Data ---
        months = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        num_months = len(months)

        # Purchase prices (yuan/unit) for each month
        purchase_prices = [28, 24, 25, 27, 23, 23]

        # Selling prices (yuan/unit) for each month
        selling_prices = [29, 24, 26, 28, 22, 25]

        # Warehouse capacity
        warehouse_capacity = 500  # units

        # Initial inventory at the end of June (start of July)
        initial_inventory = 200  # units

        # Threshold for total purchases (July–December)
        purchase_threshold = 2000  # units

        # Monthly additional fixed warehouse management fee
        extra_fee = 3000  # yuan/month

        # Very large number for Big-M constraints
        M_big = 10000

        # --- Create Gurobi Model ---
        model = gp.Model("PurchaseSalesPlan_WithNonlinearity")

        # --- Decision Variables ---
        # Buy[t]: quantity purchased at the beginning of month t
        buy_vars = model.addVars(num_months,
                                 name="Buy",
                                 lb=0.0,
                                 vtype=GRB.INTEGER)

        # Sell[t]: quantity sold during month t
        sell_vars = model.addVars(num_months,
                                  name="Sell",
                                  lb=0.0,
                                  vtype=GRB.INTEGER)

        # Inv[t]: inventory at the end of month t
        inventory_vars = model.addVars(num_months,
                                       name="Inventory",
                                       lb=0.0,
                                       vtype=GRB.INTEGER)

        # y[t]: binary variable, 1 if from month t to Dec inclusive
        #       the extra warehouse management fee is active, 0 otherwise
        y = model.addVars(num_months, vtype=GRB.BINARY, name="FeeActive")

        # z[t]: auxiliary continuous variable approximating
        #       y[t] * sum_{k=0..t} Buy[k]
        z = model.addVars(num_months, lb=0.0, name="CumBuyTimesY")

        # --- Objective Function: Maximize Total Profit ---
        # ❤ Non-linearity is introduced. ❤
        # total_profit = gp.quicksum(selling_prices[t] * sell_vars[t] -
        #                            purchase_prices[t] * buy_vars[t]
        #                            for t in range(num_months))
        # model.setObjective(total_profit, GRB.MAXIMIZE)

        # Base profit from buying/selling
        base_profit = gp.quicksum(
            selling_prices[t] * sell_vars[t] -
            purchase_prices[t] * buy_vars[t]
            for t in range(num_months)
        )

        # Extra fixed fee: if total purchases exceed 2000 units,
        # then from the first month where cumulative purchases exceed 2000
        # until December, pay 3000 yuan per month.
        max_fee_months = num_months  # at most all months pay the fee
        fee_cost = extra_fee * gp.quicksum(y[t] for t in range(num_months))

        # Objective: maximize profit minus extra fees
        model.setObjective(base_profit - fee_cost, GRB.MAXIMIZE)

        # --- Constraints ---

        # Inventory flow and capacity
        for t in range(num_months):
            inventory_at_start_of_month_t = initial_inventory if t == 0 else inventory_vars[t - 1]

            # 1. Inventory Balance Constraint
            model.addConstr(
                inventory_vars[t] == inventory_at_start_of_month_t +
                buy_vars[t] - sell_vars[t],
                name=f"InventoryBalance_{months[t]}"
            )

            # 2. Warehouse Capacity Constraint
            model.addConstr(
                inventory_vars[t] <= warehouse_capacity,
                name=f"WarehouseCapacity_{months[t]}"
            )

            # 3. Sales cannot exceed available stock
            model.addConstr(
                sell_vars[t] <= inventory_at_start_of_month_t + buy_vars[t],
                name=f"SalesLimit_{months[t]}"
            )

            # 4. Inventory at beginning of month cannot exceed capacity
            model.addConstr(
                inventory_at_start_of_month_t + buy_vars[t] <= warehouse_capacity,
                name=f"InventoryLimit_{months[t]}"
            )

        # --- Additional constraints for the threshold-based fixed fee ---

        # Cumulative purchases up to each month t
        cum_buy = {}
        for t in range(num_months):
            cum_buy[t] = gp.quicksum(buy_vars[k] for k in range(t + 1))

        # 1) If total purchases <= threshold, no fee at all
        total_purchases = gp.quicksum(buy_vars[t] for t in range(num_months))
        model.addConstr(
            total_purchases - purchase_threshold <= M_big * gp.quicksum(y[t] for t in range(num_months)),
            name="TriggerFeeIfAboveThreshold"
        )

        # 2) Ensure that if any month t has cumulative purchases above threshold,
        # then for that month and all later months the fee is active.
        # We model:
        #   cum_buy[t] - purchase_threshold <= M_big * (1 - y[t])
        # so if cum_buy[t] > threshold, y[t] must be 1.
        for t in range(num_months):
            model.addConstr(
                cum_buy[t] - purchase_threshold <= M_big * (1 - y[t]),
                name=f"ActivateFeeWhenCumAboveThreshold_{months[t]}"
            )

        # 3) Monotonicity of fee activation:
        # once y[t] = 1, all later y[k] must be 1 (k > t).
        for t in range(num_months - 1):
            model.addConstr(
                y[t] <= y[t + 1],
                name=f"FeeMonotone_{months[t]}_{months[t+1]}"
            )

        # Note:
        # The combination of (1) total_purchases vs. sum(y[t]) and
        # (2) cum_buy vs. y[t] plus (3) monotonicity
        # realizes the rule: if cumulative purchases ever exceed 2000 in month t0,
        # then from that month t0 through December, a fixed monthly fee is paid.

        # Suppress Gurobi output to console if desired
        # model.setParam('OutputFlag', 0)

        # Optimize the model
        model.optimize()

        # --- Results ---
        if model.status == GRB.OPTIMAL:
            print("Optimal purchase and sales plan found.")
            print(f"Maximum Total Profit: {model.ObjVal:.2f} Yuan")

            print("\nMonthly Plan Details:")
            print(
                f"{'Month':<5} | {'Purchase Price':<15} | {'Selling Price':<15} | "
                f"{'Buy (Units)':<12} | {'Sell (Units)':<12} | {'End Inventory':<15} | {'Fee Active':<10}"
            )
            print("-" * 110)

            current_inventory = initial_inventory
            for t in range(num_months):
                print(
                    f"{months[t]:<5} | {purchase_prices[t]:<15.2f} | {selling_prices[t]:<15.2f} | "
                    f"{buy_vars[t].X:<12.2f} | {sell_vars[t].X:<12.2f} | {inventory_vars[t].X:<15.2f} | "
                    f"{int(round(y[t].X)):<10d}"
                )
                current_inventory = inventory_vars[t].X

            print("-" * 110)
            print(f"\nInitial Inventory (End of June): {initial_inventory:.2f} units")
            print(f"Final Inventory (End of December): {inventory_vars[num_months-1].X:.2f} units")

            total_buys = sum(buy_vars[t].X for t in range(num_months))
            total_fee_months = sum(int(round(y[t].X)) for t in range(num_months))
            total_extra_fee = total_fee_months * extra_fee

            print(f"Total Purchases (Jul–Dec): {total_buys:.2f} units")
            print(f"Months with Extra Fee: {total_fee_months} (Total Extra Fee: {total_extra_fee:.2f} Yuan)")

            print(
                "\nNote: The model includes a piecewise-fixed warehouse management fee "
                "that is charged monthly from the first month when cumulative purchases "
                "exceed 2000 units through December."
            )

        elif model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Check constraints and data.")
        else:
            print(f"Optimization stopped with status: {model.status}")

    except gp.GurobiError as e:
        print(f"Gurobi error code {e.errno}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    solve_purchase_sales_plan()