import gurobipy as gp
from gurobipy import GRB
import math


def solve_purchase_sales_plan():
    """
    Solves the purchase and sales planning problem to maximize total profit
    over a 6-month period, subject to (now non-linear) warehouse capacity
    and inventory flow.
    """
    try:
        # --- Data ---
        months = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        num_months = len(months)

        # Purchase prices (yuan/unit) for each month
        purchase_prices = [28, 24, 25, 27, 23, 23]

        # Selling prices (yuan/unit) for each month
        selling_prices = [29, 24, 26, 28, 22, 25]

        # Nominal warehouse capacity
        warehouse_capacity = 500  # units

        # Initial inventory at the end of June (start of July)
        initial_inventory = 200  # units

        # --- Create Gurobi Model ---
        # Enable nonconvex handling because we will introduce cos() with a bilinear argument
        model = gp.Model("PurchaseSalesPlan_Nonlinear")
        model.Params.NonConvex = 2

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
                                       vtype=GRB.CONTINUOUS)

        # --- Auxiliary variables for non-linear capacity restriction ---
        # t_index runs from 1 to 6 for Jul..Dec
        t_index = {m: (i + 1) for i, m in enumerate(months)}  # Jul->1, ..., Dec->6

        # z_t = x_t * t   (bilinear term)
        z_vars = model.addVars(num_months,
                               name="z_xt",
                               lb=0.0,
                               vtype=GRB.CONTINUOUS)

        # arg_t = 3.1416 * z_t / 50
        arg_vars = model.addVars(num_months,
                                 name="arg_cos",
                                 lb=-GRB.INFINITY,
                                 vtype=GRB.CONTINUOUS)

        # cos_arg_t = cos(arg_t)
        cos_vars = model.addVars(num_months,
                                 name="cos_val",
                                 lb=-1.0,
                                 ub=1.0,
                                 vtype=GRB.CONTINUOUS)

        # cap_t = 500 * (1 + 0.1 * cos_arg_t)
        cap_vars = model.addVars(num_months,
                                 name="Cap",
                                 lb=0.0,
                                 vtype=GRB.CONTINUOUS)

        # --- Objective Function: Maximize Total Profit ---
        total_profit = gp.quicksum(selling_prices[t] * sell_vars[t] -
                                   purchase_prices[t] * buy_vars[t]
                                   for t in range(num_months))
        model.setObjective(total_profit, GRB.MAXIMIZE)

        # --- Constraints ---
        for t in range(num_months):
            # Inventory at the start of the current month t
            inventory_at_start_of_month_t = initial_inventory if t == 0 else inventory_vars[t - 1]

            # 1. Inventory Balance Constraint
            model.addConstr(
                inventory_vars[t] == inventory_at_start_of_month_t +
                buy_vars[t] - sell_vars[t],
                name=f"InventoryBalance_{months[t]}"
            )

            # ❤ Non-linearity is introduced. ❤
            # model.addConstr(inventory_vars[t] <= warehouse_capacity,
            #                 name=f"WarehouseCapacity_{months[t]}")

            # --- New non-linear, time- and level-dependent capacity constraint ---
            # z_t = x_t * t_index
            model.addConstr(
                z_vars[t] == inventory_vars[t] * t_index[months[t]],
                name=f"xt_times_t_{months[t]}"
            )

            # arg_t = 3.1416 * z_t / 50
            model.addConstr(
                arg_vars[t] == 3.1416 * z_vars[t] / 50.0,
                name=f"arg_cos_def_{months[t]}"
            )

            # cos_arg_t = cos(arg_t)
            model.addGenConstrCos(
                arg_vars[t],
                cos_vars[t],
                name=f"cos_constr_{months[t]}"
            )

            # cap_t = 500 * (1 + 0.1 * cos_arg_t)
            model.addConstr(
                cap_vars[t] == warehouse_capacity * (1.0 + 0.1 * cos_vars[t]),
                name=f"cap_def_{months[t]}"
            )

            # x_t <= cap_t
            model.addConstr(
                inventory_vars[t] <= cap_vars[t],
                name=f"NonlinearCapacity_{months[t]}"
            )

            # 3. Sales Constraint
            model.addConstr(
                sell_vars[t] <= inventory_at_start_of_month_t + buy_vars[t],
                name=f"SalesLimit_{months[t]}"
            )

            # ❤ Non-linearity is introduced. ❤
            # model.addConstr(inventory_at_start_of_month_t + buy_vars[t] <= warehouse_capacity,
            #                 name=f"InventoryLimit_{months[t]}")

            # Replace the above with a limit that uses the *same* effective capacity
            # for the inventory level that will result after buying and selling.
            # Here we take a conservative linear bound by ensuring the starting
            # inventory plus purchase does not exceed the maximal possible
            # effective capacity: 500 * (1 + 0.1) = 550.
            model.addConstr(
                inventory_at_start_of_month_t + buy_vars[t] <= warehouse_capacity * 1.1,
                name=f"InventoryLimit_NonlinearSafe_{months[t]}"
            )

        # Optimize the model
        model.optimize()

        # --- Results ---
        if model.status == GRB.OPTIMAL:
            print("Optimal purchase and sales plan found (non-linear capacity).")
            print(f"Maximum Total Profit: {model.ObjVal:.2f} Yuan")

            print("\nMonthly Plan Details:")
            print(
                f"{'Month':<5} | {'Purchase Price':<15} | {'Selling Price':<15} | "
                f"{'Buy (Units)':<12} | {'Sell (Units)':<12} | {'End Inventory':<15} | {'Cap (Eff.)':<12}"
            )
            print("-" * 110)

            current_inventory = initial_inventory
            for t in range(num_months):
                print(
                    f"{months[t]:<5} | {purchase_prices[t]:<15.2f} | {selling_prices[t]:<15.2f} | "
                    f"{buy_vars[t].X:<12.2f} | {sell_vars[t].X:<12.2f} | "
                    f"{inventory_vars[t].X:<15.2f} | {cap_vars[t].X:<12.2f}"
                )
                current_inventory = inventory_vars[t].X

            print("-" * 110)
            print(f"\nInitial Inventory (End of June): {initial_inventory:.2f} units")
            print(f"Final Inventory (End of December): {inventory_vars[num_months-1].X:.2f} units")

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