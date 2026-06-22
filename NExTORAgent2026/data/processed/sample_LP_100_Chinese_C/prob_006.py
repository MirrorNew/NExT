import gurobipy as gp
from gurobipy import GRB


def solve_production_planning_with_overtime():
    """
    Solves the production planning problem to maximize net profit,
    considering resource constraints, overtime pay, and a fixed
    depreciation cost if total production exceeds 15 units.

    Primary objective: maximize net profit.
    Secondary objective: among all max-profit plans, minimize overtime.
    """
    try:
        # --- Parameters ---
        products = ['A', 'B']

        # Profit per unit (excluding worker overtime pay) (yuan/unit)
        gross_profit = {'A': 5000, 'B': 11000}

        # Resource requirements per unit
        steel_req = {'A': 6, 'B': 12}     # kg/unit
        aluminum_req = {'A': 8, 'B': 20}  # kg/unit
        labor_req = {'A': 11, 'B': 24}    # hours/unit

        # Resource availability
        avail_steel = 200          # kg
        avail_aluminum = 300       # kg
        avail_labor_regular = 300  # hours

        # Overtime pay (yuan/hour)
        overtime_pay_per_hour = 100

        # Fixed equipment accelerated depreciation cost
        fixed_depreciation_cost = 2000

        # Threshold for total production quantity
        production_threshold = 15

        # --- Create Gurobi Model ---
        model = gp.Model("ProductionPlanningOvertimeWithFixedCost")

        # --- Decision Variables ---
        # X[p]: Number of units of product p to produce
        X = model.addVars(products, name="Produce", vtype=GRB.INTEGER, lb=0)

        # OT: Total overtime hours used
        OT = model.addVar(name="OvertimeHours", lb=0.0, vtype=GRB.CONTINUOUS)

        # ❤ Non-linearity is introduced. ❤
        # dep_flag: binary variable indicating if total production exceeds threshold
        dep_flag = model.addVar(name="DepreciationFlag", vtype=GRB.BINARY)

        # --- Objective Function: Maximize Net Profit (primary) ---
        total_gross_profit = gp.quicksum(gross_profit[p] * X[p]
                                         for p in products)
        total_overtime_cost = overtime_pay_per_hour * OT

        # ❤ Non-linearity is introduced. ❤
        # Add fixed depreciation cost if total production exceeds threshold
        total_depreciation_cost = fixed_depreciation_cost * dep_flag

        # Primary objective: maximize net profit
        model.setObjective(total_gross_profit
                           - total_overtime_cost
                           - total_depreciation_cost,
                           sense=GRB.MAXIMIZE)

        # --- Constraints ---
        # 1. Steel Constraint
        model.addConstr(gp.quicksum(steel_req[p] * X[p] for p in products)
                        <= avail_steel,
                        name="SteelLimit")

        # 2. Aluminum Constraint
        model.addConstr(gp.quicksum(aluminum_req[p] * X[p] for p in products)
                        <= avail_aluminum,
                        name="AluminumLimit")

        # 3. Labor Constraint
        total_labor_needed = gp.quicksum(labor_req[p] * X[p] for p in products)
        model.addConstr(total_labor_needed <= avail_labor_regular + OT,
                        name="LaborAvailability")

        # 4. Link production quantity with depreciation flag (big-M logic)
        total_production = gp.quicksum(X[p] for p in products)

        # We want:
        # if total_production > 15  => dep_flag = 1
        # if total_production <= 15 => dep_flag = 0
        #
        # Using big-M linearization with integer X and binary dep_flag:
        # total_production <= 15 + M * dep_flag
        # total_production >= 16 * dep_flag
        # Here M is an upper bound on total production; we take a safe large value.
        M = 1000

        # ❤ Non-linearity is introduced. ❤
        model.addConstr(total_production <= production_threshold + M * dep_flag,
                        name="DepreciationUpperLink")

        # ❤ Non-linearity is introduced. ❤
        model.addConstr(total_production >= (production_threshold + 1) * dep_flag,
                        name="DepreciationLowerLink")

        # At this point, any optimal solution will choose dep_flag such that
        # the fixed cost is paid iff total production > 15.

        # --- Primary optimization: maximize profit ---
        model.optimize()

        if model.status != GRB.OPTIMAL:
            print(f"Optimization stopped with status: {model.status}")
            if model.SolCount == 0:
                print("No feasible solution found.")
            return

        # Store optimal profit for secondary optimization
        optimal_profit = model.ObjVal

        # --- Secondary objective: minimize overtime under optimal profit ---
        # We now add a constraint fixing profit at its optimal value
        # (within a small tolerance) and then minimize OT.
        profit_expr = (total_gross_profit
                       - total_overtime_cost
                       - total_depreciation_cost)

        # Allow tiny numerical tolerance
        tol = 1e-4
        model.addConstr(profit_expr >= optimal_profit - tol,
                        name="FixOptimalProfit")

        # ❤ Non-linearity is introduced. ❤
        # Now change objective to minimize overtime
        model.setObjective(OT, sense=GRB.MINIMIZE)

        model.optimize()

        # --- Results ---
        if model.status == GRB.OPTIMAL:
            print("Optimal production plan found.")
            print(f"Maximum Net Profit: {optimal_profit:.2f} Yuan")

            print("\nOptimal Production Quantities (units):")
            for p in products:
                print(f"  Product {p}: {X[p].X:.0f} units")

            print(f"\nOvertime Hours Used (min among max-profit plans): {OT.X:.2f} hours")
            print(
                f"Cost of Overtime: {(overtime_pay_per_hour * OT.X):.2f} Yuan"
            )

            print("\nFixed Depreciation Cost:")
            print(f"  Depreciation Flag (1 = pay, 0 = no pay): {dep_flag.X:.0f}")
            print(
                f"  Depreciation Cost Paid: "
                f"{(fixed_depreciation_cost * dep_flag.X):.2f} Yuan"
            )

            print("\nResource Utilization:")
            steel_used = sum(steel_req[p] * X[p].X for p in products)
            aluminum_used = sum(aluminum_req[p] * X[p].X for p in products)
            labor_needed_val = sum(labor_req[p] * X[p].X for p in products)
            total_prod_val = sum(X[p].X for p in products)

            print(
                f"  Steel Used: {steel_used:.2f} / {avail_steel} kg "
                f"({(steel_used / avail_steel * 100) if avail_steel > 0 else 0:.1f}%)"
            )
            print(
                f"  Aluminum Used: {aluminum_used:.2f} / {avail_aluminum} kg "
                f"({(aluminum_used / avail_aluminum * 100) if avail_aluminum > 0 else 0:.1f}%)"
            )
            print(f"  Total Labor Needed: {labor_needed_val:.2f} hours")
            print(
                f"    Met by Regular Hours: "
                f"{min(labor_needed_val, avail_labor_regular):.2f} / {avail_labor_regular} hours"
            )
            if OT.X > 1e-6:
                print(f"    Met by Overtime Hours: {OT.X:.2f} hours")

            print(f"\nTotal Production Quantity: {total_prod_val:.0f} units")
            print(
                "  Threshold exceeded (production > 15): "
                f"{'Yes' if total_prod_val > 15 + 1e-6 else 'No'}"
            )

        elif model.status == GRB.INFEASIBLE:
            print(
                "Model is infeasible. Check constraints and resource availability."
            )
        else:
            print(f"Optimization stopped with status: {model.status}")
            if model.SolCount == 0:
                print("No feasible solution found.")

    except gp.GurobiError as e:
        print(f"Gurobi error code {e.errno}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    solve_production_planning_with_overtime()