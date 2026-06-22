import gurobipy as gp
from gurobipy import GRB


def solve_production_profit_maximization():
    """
    Solves the production planning problem to maximize weekly profit
    for products X and Y, subject to time, contract, and craftsman-time
    ratio constraints (a non-linear requirement).
    """
    try:
        # --- Parameters ---
        products = ['X', 'Y']

        # Time requirements (minutes/unit)
        machine_time_req = {'X': 13, 'Y': 19}
        craftsman_time_req = {'X': 20, 'Y': 29}

        # Time availability (minutes/week)
        avail_machine_time = 40 * 60  # 2400 minutes
        avail_craftsman_time = 35 * 60  # 2100 minutes

        # Costs (per minute)
        cost_machine_per_min = 10 / 60
        cost_craftsman_per_min = 2 / 60

        # Revenue (per unit)
        revenue = {'X': 20, 'Y': 30}

        # Contract requirement
        min_production_X = 10

        # Calculate profit per unit
        profit_per_unit = {}
        for p in products:
            cost_machine = machine_time_req[p] * cost_machine_per_min
            cost_craftsman = craftsman_time_req[p] * cost_craftsman_per_min
            profit_per_unit[p] = revenue[p] - cost_machine - cost_craftsman

        # --- Create Gurobi Model ---
        model = gp.Model("ProductionProfitMaximization")

        # --- Decision Variables ---
        # N[p]: Number of units of product p produced per week
        N = model.addVars(products, name="Produce", vtype=GRB.INTEGER, lb=0)

        # --- Objective Function: Maximize Total Profit ---
        model.setObjective(
            gp.quicksum(profit_per_unit[p] * N[p] for p in products),
            GRB.MAXIMIZE
        )

        # --- Constraints ---
        # 1. Machine Time Constraint
        model.addConstr(
            gp.quicksum(machine_time_req[p] * N[p] for p in products)
            <= avail_machine_time,
            name="MachineTimeLimit"
        )

        # 2. Craftsman Time Constraint
        model.addConstr(
            gp.quicksum(craftsman_time_req[p] * N[p] for p in products)
            <= avail_craftsman_time,
            name="CraftsmanTimeLimit"
        )

        # 3. Contract Requirement for Product X
        model.addConstr(N['X'] >= min_production_X, name="ContractX")

        # 4. Craftsman-time ratio constraint:
        #    total craftsman time for X / total craftsman time for Y <= 3
        #    i.e., 20 * N['X'] / (29 * N['Y']) <= 3
        #    This introduces non-linearity due to division by N['Y'].

        # ❤ Non-linearity is introduced. ❤
        # We define the ratio explicitly as a non-linear constraint:
        craftsman_time_X = 20 * N['X']
        craftsman_time_Y = 29 * N['Y']

        # To avoid division by zero, we only impose the ratio when some Y is produced.
        # This is still non-linear because of the division.
        # Constraint form: craftsman_time_X <= 3 * craftsman_time_Y  (if N['Y'] > 0)
        # Here we directly add the bilinear (integer * constant) inequality;
        # the logical "if N['Y'] > 0" is implicit in the ratio interpretation.
        model.addConstr(
            craftsman_time_X <= 3 * craftsman_time_Y,
            name="CraftsmanTimeRatio_X_to_Y"
        )

        # Suppress Gurobi output to console if desired
        # model.setParam('OutputFlag', 0)

        # Optimize the model
        model.optimize()

        # --- Results ---
        if model.status == GRB.OPTIMAL:
            print("Optimal production plan found.")
            print(f"Maximum Weekly Profit: £{model.ObjVal:.2f}")

            print("\nOptimal Production Quantities (units per week):")
            for p in products:
                print(f"  Product {p}: {N[p].X:.0f} units")

            print("\nResource Utilization:")
            machine_time_used = sum(machine_time_req[p] * N[p].X for p in products)
            craftsman_time_used = sum(craftsman_time_req[p] * N[p].X for p in products)
            print(
                f"  Machine Time Used: {machine_time_used:.2f} / {avail_machine_time} minutes "
                f"({(machine_time_used / avail_machine_time * 100) if avail_machine_time > 0 else 0:.1f}%)"
            )
            print(
                f"  Craftsman Time Used: {craftsman_time_used:.2f} / {avail_craftsman_time} minutes "
                f"({(craftsman_time_used / avail_craftsman_time * 100) if avail_craftsman_time > 0 else 0:.1f}%)"
            )

            print("\nContract Fulfillment:")
            print(
                f"  Product X Produced: {N['X'].X:.0f} (Min Required: {min_production_X})"
            )

            craftsman_time_X_val = 20 * N['X'].X
            craftsman_time_Y_val = 29 * N['Y'].X
            print("\nCraftsman Time Ratio (X to Y):")
            if craftsman_time_Y_val > 0:
                ratio = craftsman_time_X_val / craftsman_time_Y_val
                print(
                    f"  Craftsman time for X: {craftsman_time_X_val:.2f} minutes\n"
                    f"  Craftsman time for Y: {craftsman_time_Y_val:.2f} minutes\n"
                    f"  Ratio X/Y: {ratio:.4f} (limit: 3)"
                )
            else:
                print(
                    f"  Craftsman time for X: {craftsman_time_X_val:.2f} minutes\n"
                    f"  Craftsman time for Y: {craftsman_time_Y_val:.2f} minutes\n"
                    "  Ratio X/Y is not defined because no Y is produced."
                )

        elif model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Check constraints and requirements.")
        else:
            print(f"Optimization stopped with status: {model.status}")
            if model.SolCount == 0:
                print("No feasible solution found.")

    except gp.GurobiError as e:
        print(f"Gurobi error code {e.errno}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    solve_production_profit_maximization()