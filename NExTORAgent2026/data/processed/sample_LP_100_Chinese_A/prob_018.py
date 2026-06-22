import math

import gurobipy as gp
from gurobipy import GRB


def solve_production_profit_maximization():
    """
    Solves the production planning problem to maximize weekly profit
    for products X and Y, subject to time and (now non-linear) contract constraints.
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

        # Base contract requirement (used in the original linear model)
        min_production_X = 10

        # Exponential growth factor for the new (non-linear) contract requirement
        growth_factor = 1.3

        # Calculate profit per unit
        profit_per_unit = {}
        for p in products:
            cost_machine = machine_time_req[p] * cost_machine_per_min
            cost_craftsman = craftsman_time_req[p] * cost_craftsman_per_min
            profit_per_unit[p] = revenue[p] - cost_machine - cost_craftsman
            # print(f"Profit per unit {p}: {profit_per_unit[p]:.4f}")  # For verification

        # --- Create Gurobi Model ---
        # Use a general (potentially non-linear) model
        model = gp.Model("ProductionProfitMaximization_Nonlinear")

        # --- Decision Variables ---
        # N[p]: Number of units of product p produced per week (integer, >= 0)
        N = model.addVars(products, name="Produce", vtype=GRB.INTEGER, lb=0)

        # --- Objective Function: Maximize Total Profit ---
        # Profit = Sum(Profit_per_unit[p] * N[p])
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
        # Original linear contract constraint:
        # model.addConstr(N['X'] >= min_production_X, name="ContractX")

        # ❤ Non-linearity is introduced. ❤
        # New non-linear contract:
        # If n = N['X'] > 10, the minimum required supply is:
        #   10 * 1.3^(n - 10)
        # and the actual production must satisfy:
        #   N['X'] >= 10 * 1.3^(N['X'] - 10)
        #
        # Note: This is an inherently non-linear, self-referential constraint.
        # We model it directly as a general non-linear constraint.
        #
        # For N['X'] <= 10, the expression on the right-hand side is <= 10,
        # so the inequality N['X'] >= 10 * 1.3^(N['X'] - 10) is automatically
        # satisfied when N['X'] >= 0; thus it is consistent with the
        # description that the basic requirement is 10 when planning to
        # produce around that level, and from the 11th unit onward the
        # lower bound grows exponentially.
        #
        # Implemented as: N['X'] >= 10 * growth_factor ** (N['X'] - 10)
        Y = model.addVar(vtype=GRB.CONTINUOUS, name="Y")
        Y1 = model.addVar(vtype=GRB.CONTINUOUS, name="Y1")
        Y2 = model.addVar(vtype=GRB.CONTINUOUS, name="Y2")

        model.addConstr(Y1 ==math.log(growth_factor) * Y2)
        model.addConstr(Y2 == N['X'] - 10)
        model.addGenConstrExp(Y1,Y)

        model.addConstr(
            N['X'] >= 10 * Y,
            name="NonlinearContractX"
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
            machine_time_used = sum(machine_time_req[p] * N[p].X
                                    for p in products)
            craftsman_time_used = sum(craftsman_time_req[p] * N[p].X
                                      for p in products)
            print(
                f"  Machine Time Used: {machine_time_used:.2f} / {avail_machine_time} minutes "
                f"({(machine_time_used / avail_machine_time * 100) if avail_machine_time > 0 else 0:.1f}%)"
            )
            print(
                f"  Craftsman Time Used: {craftsman_time_used:.2f} / {avail_craftsman_time} minutes "
                f"({(craftsman_time_used / avail_craftsman_time * 100) if avail_craftsman_time > 0 else 0:.1f}%)"
            )

            print("\nContract Fulfillment (Non-linear):")
            print(f"  Product X Produced: {N['X'].X:.0f}")
            # Evaluate the RHS of the non-linear contract for reporting
            n_x = N['X'].X
            required_min_x = 10 * (growth_factor ** (n_x - 10))
            print(f"  Non-linear minimum required (given n = {n_x:.0f}): {required_min_x:.4f}")

        elif model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Check constraints and requirements.")
            # Optionally, compute and print IIS (Irreducible Inconsistent Subsystem)
            # model.computeIIS()
            # model.write("production_profit_iis.ilp")
            # print("IIS written to production_profit_iis.ilp.")
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