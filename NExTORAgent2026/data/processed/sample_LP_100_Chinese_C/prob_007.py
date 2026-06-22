import gurobipy as gp
from gurobipy import GRB


def solve_complete_production_planning():
    """
    Solves the production planning problem with fixed activation costs,
    minimum batch sizes, a shared time resource constraint, and an
    additional nonlinear-type (piecewise) condition on overtime activation:
    If total monthly production of a product (in 100kg units) exceeds 4000,
    an extra one-time overtime activation cost of 30000 USD is incurred.
    """
    try:
        # --- 1. Parameters ---
        products = ['A1', 'A2', 'A3']

        max_demand = {'A1': 5300, 'A2': 4500, 'A3': 5400}  # units of 100kg
        selling_price = {'A1': 124, 'A2': 109, 'A3': 115}  # $/100kg
        production_cost = {'A1': 73.30, 'A2': 52.90, 'A3': 65.40}  # $/100kg
        production_quota = {'A1': 500, 'A2': 450, 'A3': 550}  # 100kg units/day

        activation_cost = {'A1': 170000, 'A2': 150000, 'A3': 100000}  # $
        min_batch = {'A1': 20, 'A2': 20, 'A3': 16}  # units of 100kg

        available_days = 22

        # Extra overtime activation cost if monthly production exceeds 4000 units (100kg each)
        overtime_threshold = 4000        # 100kg units
        overtime_activation_cost = 30000  # $

        # Pre-calculate profit per unit for clarity
        profit_per_unit = {
            p: selling_price[p] - production_cost[p]
            for p in products
        }

        # --- 2. Model Initialization ---
        model = gp.Model("CompleteProductionPlanning_WithOvertimeTrigger")

        # --- 3. Decision Variables ---
        # produce_qty[p]: quantity of product p to produce (in 100kg units)
        produce_qty = model.addVars(products,
                                    vtype=GRB.INTEGER,
                                    name="produce_qty",
                                    lb=0)

        # use_product[p]: binary variable, 1 if product p is produced, 0 otherwise
        use_product = model.addVars(products,
                                    vtype=GRB.BINARY,
                                    name="use_product")

        # ❤ Non-linearity is introduced. ❤
        # overtime_used[p]: binary variable, 1 if total production of p exceeds overtime_threshold, 0 otherwise
        overtime_used = model.addVars(products,
                                      vtype=GRB.BINARY,
                                      name="overtime_used")

        # --- 4. Objective Function ---
        # Variable profit from units produced
        total_variable_profit = gp.quicksum(profit_per_unit[p] * produce_qty[p]
                                            for p in products)

        # Conditional activation costs (fixed setup)
        total_activation_cost = gp.quicksum(activation_cost[p] * use_product[p]
                                            for p in products)

        # ❤ Non-linearity is introduced. ❤
        # Additional overtime activation costs when production exceeds threshold
        total_overtime_cost = gp.quicksum(overtime_activation_cost *
                                          overtime_used[p]
                                          for p in products)

        # Maximize total profit including overtime trigger cost
        # ❤ Non-linearity is introduced. ❤
        # model.setObjective(total_variable_profit - total_activation_cost,
        #                    GRB.MAXIMIZE)
        model.setObjective(
            total_variable_profit - total_activation_cost - total_overtime_cost,
            GRB.MAXIMIZE
        )

        # --- 5. Constraints ---

        # Constraint 5.1: Shared Resource Constraint (Total Production Time)
        model.addConstr(
            gp.quicksum(produce_qty[p] / production_quota[p] for p in products)
            <= available_days, "TotalProductionTime")

        # Constraints applied to each product individually
        for p in products:
            # Constraint 5.2: Maximum Demand Constraint
            model.addConstr(produce_qty[p] <= max_demand[p], f"MaxDemand_{p}")

            # Constraint 5.3: Minimum batch size if product is used
            model.addConstr(produce_qty[p] >= min_batch[p] * use_product[p],
                            f"MinBatchLink_{p}")

            # Constraint 5.4: Link binary activation to upper bound on production
            model.addConstr(produce_qty[p] <= max_demand[p] * use_product[p],
                            f"ActivationLink_{p}")

            # --- New logic for “exceed 4000 units → extra 30000 cost” ---

            # ❤ Non-linearity is introduced. ❤
            # If overtime_used[p] = 1, production must be strictly above threshold.
            # If overtime_used[p] = 0, production must be <= threshold.
            # Implemented via big-M style linearization.

            # Big-M: maximum possible production for product p
            M_p = max_demand[p]

            # 1) If overtime_used[p] = 0, this enforces produce_qty[p] <= overtime_threshold
            #    If overtime_used[p] = 1, the RHS becomes overtime_threshold + M_p (non-binding).
            model.addConstr(
                produce_qty[p] <= overtime_threshold + M_p * overtime_used[p],
                f"OvertimeUpper_{p}"
            )

            # 2) If overtime_used[p] = 1, this forces produce_qty[p] >= overtime_threshold + 1
            #    If overtime_used[p] = 0, RHS = overtime_threshold + 1, but this is relaxed
            #    using (use_product[p]) so that if the product is not produced,
            #    this constraint does not artificially require positive production.
            #    We ensure: produce_qty[p] >= (overtime_threshold + 1) * overtime_used[p]
            model.addConstr(
                produce_qty[p] >= (overtime_threshold + 1) * overtime_used[p],
                f"OvertimeLower_{p}"
            )

            # 3) Logical consistency: if product is not used at all, then overtime cannot be triggered
            model.addConstr(
                overtime_used[p] <= use_product[p],
                f"OvertimeUseLink_{p}"
            )

        # --- 6. Optimize Model ---
        model.optimize()

        # --- 7. Results ---
        print("-" * 50)
        if model.status == GRB.OPTIMAL:
            print("Optimal production plan found!")
            print(f"Maximum Total Profit: ${model.objVal:,.2f}")
            print("-" * 50)
            print("Production Details:")
            total_days_used = 0
            for p in products:
                qty = produce_qty[p].X
                days_for_p = qty / production_quota[p]
                total_days_used += days_for_p

                print(f"  Product {p}:")
                if use_product[p].X > 0.5:
                    print(f"    Status: PRODUCED")
                    print(f"    Produce Quantity (100kg units): {qty:.0f}")
                    print(f"    Days Used: {days_for_p:.2f} days")

                    # Base net profit excluding overtime
                    net_profit_p = (profit_per_unit[p] * qty) - activation_cost[p]

                    # Subtract overtime cost if incurred
                    if overtime_used[p].X > 0.5:
                        print("    Overtime threshold exceeded: YES")
                        print(f"    Overtime activation cost: ${overtime_activation_cost:,.2f}")
                        net_profit_p -= overtime_activation_cost
                    else:
                        print("    Overtime threshold exceeded: NO")
                        print(f"    Overtime activation cost: $0.00")

                    print(
                        f"    Net Profit (incl. all activation costs): ${net_profit_p:,.2f}"
                    )
                else:
                    print(f"    Status: NOT PRODUCED")
                    print("    Overtime threshold exceeded: NO")
                    print("    Overtime activation cost: $0.00")
                print("-" * 25)

            print("Overall Resource Utilization:")
            print(
                f"  Total Production Days Used: {total_days_used:.2f} / {available_days} days"
            )
            print("-" * 50)

        elif model.status == GRB.INFEASIBLE:
            print(
                "Model is infeasible. No solution exists that satisfies all constraints."
            )
        else:
            print(f"Optimization was stopped with status: {model.status}")

    except gp.GurobiError as e:
        print(f"Gurobi error code {e.errno}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    solve_complete_production_planning()