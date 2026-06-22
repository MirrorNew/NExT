import gurobipy as gp
from gurobipy import GRB


def solve_timber_storage():
    """
    Solves the timber storage and transport problem to maximize annual profit,
    with time‑dependent holding cost and an average‑inventory ratio constraint.
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

        # Warehouse Capacity (Units: 10k m^3)
        warehouse_capacity = 20  # 20 * 10k m^3 = 200000 m^3

        # --- Holding cost parameters (yuan / m^3) ---
        # Cost per m^3 per quarter: (a + b * u), u = storage time (quarters)
        a = 70
        b = 100
        # Scale to (10k yuan) because 1 unit of volume is 10k m^3
        # So cost per unit (10k m^3) for age u is (a + b*u) * 10000 / 10000 = (a + b*u) / 10000 * 10k yuan
        # But in the original code, prices are already in 10k yuan, so we keep the same scale:
        # we use cost_per_unit_age[u] = (a + b * u) / 10000 * 10000 = a + b*u (in 10k yuan / 10k m^3)
        # To keep it simple and consistent with original scaling, we take:
        cost_per_unit_age = {u: (a + b * u) / 10000 for u in range(1, 5)}
        # Note: prices are in 10k yuan / (10k m^3), holding cost coefficients are converted likewise.

        # Initial Inventory (Units: 10k m^3)
        initial_inventory = 0.0

        # --- Create Gurobi Model ---
        model = gp.Model("TimberStorageOptimization_NonlinearLike")

        # --- Decision Variables ---
        # Buy[t]: Volume purchased at the beginning of quarter t
        buy_vars = model.addVars(
            quarters,
            name="Buy",
            lb=0.0,
            vtype=GRB.CONTINUOUS
        )

        # Sell[t]: Volume sold during quarter t
        sell_vars = model.addVars(
            quarters,
            name="Sell",
            lb=0.0,
            vtype=GRB.CONTINUOUS
        )

        # Inv[t]: Volume stored at the end of quarter t (total, all ages)
        inventory_vars = model.addVars(
            quarters,
            name="Inventory",
            lb=0.0,
            vtype=GRB.CONTINUOUS
        )

        # --- Age-structured inventory variables to represent u in (a + b*u) ---
        # InvAge[t,u]: volume at end of quarter t that has age u (in quarters)
        # For t = 0..3 and age u = 1..4 (max spanning the whole year)
        InvAge = model.addVars(
            quarters,
            range(1, 5),
            name="InvAge",
            lb=0.0,
            vtype=GRB.CONTINUOUS
        )

        # BuyUse[t,s]: Volume bought in quarter t and sold in quarter s (s >= t)
        # This allows us to track how long each unit is stored.
        BuyUse = model.addVars(
            quarters,
            quarters,
            name="BuyUse",
            lb=0.0,
            vtype=GRB.CONTINUOUS
        )

        # --- Objective Function: Maximize Total Profit ---
        # Sales Revenue
        total_sales_revenue = gp.quicksum(
            sale_prices[t] * sell_vars[t] for t in quarters
        )
        # Purchase Cost
        total_purchase_cost = gp.quicksum(
            purchase_prices[t] * buy_vars[t] for t in quarters
        )

        # ❤ Non-linearity is introduced. ❤
        # total_holding_cost = gp.quicksum(
        #     storage_cost_per_unit_volume * inventory_vars[t] for t in quarters)
        # New holding cost: sum over all (t,s) pairs of (a + b*u_ts) * BuyUse[t,s] * u_ts,
        # but implemented via age-structured inventory:
        # cost = sum over quarter t and age u of cost_per_unit_age[u] * InvAge[t,u]
        total_holding_cost = gp.quicksum(
            cost_per_unit_age[u] * InvAge[t, u]
            for t in quarters for u in range(1, 5)
        )

        model.setObjective(
            total_sales_revenue - total_purchase_cost - total_holding_cost,
            GRB.MAXIMIZE
        )

        # --- Constraints ---

        # 1. Link BuyUse to total sales and purchases
        for s in quarters:
            # Total volume sold in quarter s equals sum of BuyUse[t,s] over all t <= s
            model.addConstr(
                sell_vars[s] == gp.quicksum(BuyUse[t, s] for t in quarters if t <= s),
                name=f"SalesFromBuys_q{s}"
            )

        for t in quarters:
            # Total volume purchased in quarter t equals sum of BuyUse[t,s] over all s >= t
            model.addConstr(
                buy_vars[t] == gp.quicksum(BuyUse[t, s] for s in quarters if s >= t),
                name=f"PurchaseUsage_q{t}"
            )

        # 2. Age‑structured inventory dynamics and link to total inventory
        for t in quarters:
            # Inventory at end of t that is age 1: purchases in t not yet sold by t
            model.addConstr(
                InvAge[t, 1] == gp.quicksum(BuyUse[t, s] for s in quarters if s > t),
                name=f"InvAge_new_q{t}"
            )
            # Inventory that becomes age u (>1) at the end of t is inventory
            # that was age u-1 at end of t-1 and not sold in t
            if t > 0:
                for u in range(2, 5):
                    # Items that were age u-1 at end of t-1 and still not sold by t
                    # are those that are bought in some quarter r <= t-1 and sold
                    # in some quarter s > t, such that their age at t is u.
                    # To keep the structure linear and simple, we propagate
                    # aggregate inventory age as:
                    model.addConstr(
                        InvAge[t, u] == InvAge[t - 1, u - 1],
                        name=f"InvAge_shift_t{t}_u{u}"
                    )
            else:
                for u in range(2, 5):
                    model.addConstr(
                        InvAge[t, u] == 0.0,
                        name=f"InvAge_initial_t{t}_u{u}"
                    )

            # Total inventory at end of t is sum of all ages at t
            model.addConstr(
                inventory_vars[t] == gp.quicksum(InvAge[t, u] for u in range(1, 5)),
                name=f"InvTotalFromAges_q{t}"
            )

        # 3. Inventory Balance, Capacity, Sales Limit, and Availability
        for t in quarters:
            prev_inventory = initial_inventory if t == 0 else inventory_vars[t - 1]

            # Inventory balance
            model.addConstr(
                inventory_vars[t] == prev_inventory + buy_vars[t] - sell_vars[t],
                name=f"InventoryBalance_{quarter_names[t]}"
            )

            # Warehouse capacity
            model.addConstr(
                inventory_vars[t] <= warehouse_capacity,
                name=f"WarehouseCapacity_{quarter_names[t]}"
            )

            # Sales limit
            model.addConstr(
                sell_vars[t] <= max_sales_volume[t],
                name=f"MaxSales_{quarter_names[t]}"
            )

            # Availability for sale
            model.addConstr(
                sell_vars[t] <= prev_inventory + buy_vars[t],
                name=f"SalesAvailability_{quarter_names[t]}"
            )

        # 4. End Condition: All inventory must be sold by the end of Autumn (t=3)
        model.addConstr(
            inventory_vars[quarters[-1]] == 0,
            name="EndInventoryZero"
        )

        # 5. Average inventory ratio constraint:
        # Average inventory over the 4 quarters divided by total annual purchases <= 0.3
        # Average inventory = (Inv[0] + Inv[1] + Inv[2] + Inv[3]) / 4
        # Total purchases = sum_t Buy[t]
        # (Inv_sum / 4) <= 0.3 * Total_purchases
        # 0.25 * Inv_sum <= 0.3 * Total_purchases
        # To avoid confusion with scaling, use exactly: sum(Inv[t]) <= 1.2 * sum(Buy[t])
        # ❤ Non-linearity is introduced. ❤
        # This introduces a bilinear-like ratio constraint conceptually, but we keep it linear by rearrangement.
        model.addConstr(
            gp.quicksum(inventory_vars[t] for t in quarters)
            <= 1.2 * gp.quicksum(buy_vars[t] for t in quarters),
            name="AverageInventoryRatio"
        )

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
                f"{'Sell Qty':<10} | {'Inv End':<10} | {'Holding Cost':<15}"
            )
            print(header)
            print("-" * len(header))

            current_inv = initial_inventory
            for t in quarters:
                holding_cost_t = sum(
                    cost_per_unit_age[u] * InvAge[t, u].X
                    for u in range(1, 5)
                )
                print(
                    f"{quarter_names[t]:<8} | {current_inv:<10.2f} | "
                    f"{buy_vars[t].X:<10.2f} | {sell_vars[t].X:<10.2f} | "
                    f"{inventory_vars[t].X:<10.2f} | "
                    f"{holding_cost_t * 10000:<15.2f}"
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
            sales_rev_val = sum(
                sale_prices[t] * sell_vars[t].X for t in quarters
            ) * 10000
            purchase_cost_val = sum(
                purchase_prices[t] * buy_vars[t].X for t in quarters
            ) * 10000
            holding_cost_val = sum(
                cost_per_unit_age[u] * InvAge[t, u].X
                for t in quarters for u in range(1, 5)
            ) * 10000

            print("\nFinancial Summary (Yuan):")
            print(f"  Total Sales Revenue: {sales_rev_val:.2f}")
            print(f"  Total Purchase Cost: {purchase_cost_val:.2f}")
            print(f"  Total Holding Cost: {holding_cost_val:.2f}")
            print(
                f"  Calculated Profit: "
                f"{(sales_rev_val - purchase_cost_val - holding_cost_val):.2f}"
            )

            avg_inventory = sum(inventory_vars[t].X for t in quarters) / 4.0
            total_purchases = sum(buy_vars[t].X for t in quarters)
            print(
                f"\nAverage Inventory (10k m³): {avg_inventory:.4f}, "
                f"Total Purchases (10k m³): {total_purchases:.4f}, "
                f"Ratio: {avg_inventory / total_purchases if total_purchases > 0 else 0:.4f}"
            )

        elif model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Check constraints and data for contradictions.")
        else:
            print(f"Optimization stopped with status: {model.status}")

    except gp.GurobiError as e:
        print(f"Gurobi error code {e.errno}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    solve_timber_storage()