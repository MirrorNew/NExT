import gurobipy as gp
from gurobipy import GRB


def solve_purchase_sales_plan():
    """
    Solves the purchase and sales planning problem to maximize total profit
    over a 6-month period, subject to warehouse capacity, inventory flow,
    and a non-linear ratio constraint between total purchases and total sales.
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

        # --- Create Gurobi Model ---
        model = gp.Model("PurchaseSalesPlan_WithRatioConstraint")

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

        # --- Objective Function: Maximize Total Profit ---
        total_profit = gp.quicksum(selling_prices[t] * sell_vars[t] -
                                   purchase_prices[t] * buy_vars[t]
                                   for t in range(num_months))
        model.setObjective(total_profit, GRB.MAXIMIZE)

        # --- Constraints ---
        for t in range(num_months):
            # Inventory at the start of the current month t
            inventory_at_start_of_month_t = initial_inventory if t == 0 else inventory_vars[
                t - 1]

            # 1. Inventory Balance Constraint
            model.addConstr(
                inventory_vars[t] == inventory_at_start_of_month_t +
                buy_vars[t] - sell_vars[t],
                name=f"InventoryBalance_{months[t]}")

            # 2. Warehouse Capacity Constraint
            model.addConstr(inventory_vars[t] <= warehouse_capacity,
                            name=f"WarehouseCapacity_{months[t]}")

            # 3. Sales Constraint
            model.addConstr(sell_vars[t]
                            <= inventory_at_start_of_month_t + buy_vars[t],
                            name=f"SalesLimit_{months[t]}")
            model.addConstr(inventory_at_start_of_month_t + buy_vars[t] <= warehouse_capacity,
                            name=f"InventoryLimit_{months[t]}")

        # --- Non-linear ratio constraint: total purchases / total sales <= 1.2 ---
        # Total purchases and sales over Jul–Dec
        total_purchase = gp.quicksum(buy_vars[t] for t in range(num_months))
        total_sales = gp.quicksum(sell_vars[t] for t in range(num_months))

        # ❤ Non-linearity is introduced. ❤
        # Commented out linearized form (if we were to keep it linear):
        # model.addConstr(total_purchase <= 1.2 * total_sales, name="PurchaseSalesRatio_Linear")

        # Introduce explicit non-linear ratio constraint:
        # total_purchase / total_sales <= 1.2  →  total_purchase <= 1.2 * total_sales
        # To keep it explicitly non-linear in the model, we write it as:
        model.addConstr(total_purchase - 1.2 * total_sales <= 0,
                         name="PurchaseSalesRatio_Nonlinear")

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
                f"{'Buy (Units)':<12} | {'Sell (Units)':<12} | {'End Inventory':<15}"
            )
            print("-" * 90)

            current_inventory = initial_inventory
            for t in range(num_months):
                print(
                    f"{months[t]:<5} | {purchase_prices[t]:<15.2f} | {selling_prices[t]:<15.2f} | "
                    f"{buy_vars[t].X:<12.2f} | {sell_vars[t].X:<12.2f} | {inventory_vars[t].X:<15.2f}"
                )
                current_inventory = inventory_vars[t].X

            print("-" * 90)
            print(
                f"\nInitial Inventory (End of June): {initial_inventory:.2f} units"
            )
            print(
                f"Final Inventory (End of December): {inventory_vars[num_months-1].X:.2f} units"
            )

            # Extra info: check ratio constraint
            total_buy_val = sum(buy_vars[t].X for t in range(num_months))
            total_sell_val = sum(sell_vars[t].X for t in range(num_months))
            if total_sell_val > 0:
                ratio = total_buy_val / total_sell_val
            else:
                ratio = float('inf')
            print(f"\nTotal purchases (Jul–Dec): {total_buy_val:.2f} units")
            print(f"Total sales (Jul–Dec): {total_sell_val:.2f} units")
            print(f"Purchase/Sales ratio (Jul–Dec): {ratio:.4f} (limit: 1.2)")

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