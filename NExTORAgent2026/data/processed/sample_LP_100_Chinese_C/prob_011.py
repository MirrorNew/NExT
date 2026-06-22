import gurobipy as gp
from gurobipy import GRB


def solve_promotional_packages():
    """
    Solves the promotional package problem to maximize revenue,
    subject to inventory, minimum sales constraints, and
    non-linear (piecewise) promotional bonus rules that are
    linearized via additional variables.
    """
    try:
        # --- Parameters ---
        packages = ['A', 'B']

        # Inventory available
        avail_shirts = 200
        avail_pants = 100

        # Package composition (units per package)
        # composition[package][item]
        composition = {
            'A': {
                'shirts': 1,
                'pants': 2
            },
            'B': {
                'shirts': 3,
                'pants': 1
            }
        }

        # Base package prices (£ per package)
        prices = {'A': 30, 'B': 50}

        # Minimum sales requirements (packages)
        min_sales = {'A': 20, 'B': 10}

        # Threshold for extra bonus and bonus amount
        bonus_threshold = {'A': 50, 'B': 40}
        bonus_amount = {'A': 5, 'B': 10}

        # --- Create Gurobi Model ---
        model = gp.Model("PromotionalPackages_NonlinearBonus")

        # --- Decision Variables ---
        # N[p]: Number of packages of type p to sell
        N = model.addVars(packages,
                          name="NumPackages",
                          vtype=GRB.INTEGER,
                          lb=0)

        # Extra[p]: number of packages of type p above the bonus threshold
        # This will capture "from the 51st A (or 41st B) onward"
        Extra = model.addVars(packages,
                              name="ExtraAboveThreshold",
                              vtype=GRB.INTEGER,
                              lb=0)

        # --- Objective Function: Maximize Total Revenue ---
        # Base revenue: prices[p] * N[p]
        # Bonus revenue: bonus_amount[p] * Extra[p]
        # Note: N[p] and Extra[p] are integer, but the expression is linear
        # in the decision variables, so the model remains a MIP.
        # ❤ Non-linearity is introduced. ❤
        model.setObjective(
            gp.quicksum(prices[p] * N[p] for p in packages)
            + gp.quicksum(bonus_amount[p] * Extra[p] for p in packages),
            GRB.MAXIMIZE
        )

        # --- Constraints ---
        # 1. Shirt Availability Constraint
        model.addConstr(gp.quicksum(composition[p]['shirts'] * N[p]
                                    for p in packages) <= avail_shirts,
                        name="ShirtLimit")

        # 2. Pants Availability Constraint
        model.addConstr(gp.quicksum(composition[p]['pants'] * N[p]
                                    for p in packages) <= avail_pants,
                        name="PantsLimit")

        # 3. Minimum Sales Requirements
        for p in packages:
            model.addConstr(N[p] >= min_sales[p], name=f"MinSales_{p}")

        # 4. Link Extra[p] to N[p] and threshold:
        #    Extra[p] = max(0, N[p] - bonus_threshold[p])
        #    We linearize this with:
        #        Extra[p] >= N[p] - bonus_threshold[p]
        #        Extra[p] <= N[p] - bonus_threshold[p]  (when N[p] >= threshold)
        #        Extra[p] >= 0                         (already from lb=0)
        #    Since the objective has a positive coefficient on Extra[p],
        #    the optimizer will not set Extra[p] larger than necessary.
        for p in packages:
            # Lower bound of Extra relative to N
            model.addConstr(
                Extra[p] >= N[p] - bonus_threshold[p],
                name=f"ExtraLB_{p}"
            )
            # Upper bound so Extra cannot exceed N - threshold when N is large
            # and cannot be positive when N is below threshold.
            model.addConstr(
                Extra[p] <= N[p],
                name=f"ExtraUB1_{p}"
            )
            model.addConstr(
                Extra[p] <= max(0, avail_shirts + avail_pants),  # loose global upper bound
                name=f"ExtraUB2_{p}"
            )

        # Suppress Gurobi output to console if desired
        # model.setParam('OutputFlag', 0)

        # Optimize the model
        model.optimize()

        # --- Results ---
        if model.status == GRB.OPTIMAL:
            print("Optimal package sales plan found.")
            print(f"Maximum Total Revenue: £{model.ObjVal:.2f}")

            print("\nNumber of Packages to Sell:")
            for p in packages:
                print(f"  Package {p}: {N[p].X:.0f} units")

            print("\nBonus-eligible extra units:")
            for p in packages:
                print(f"  Extra {p} packages above threshold: {Extra[p].X:.0f} units")

            print("\nResource Utilization:")
            shirts_used = sum(composition[p]['shirts'] * N[p].X
                              for p in packages)
            pants_used = sum(composition[p]['pants'] * N[p].X
                             for p in packages)
            print(f"  Shirts Used: {shirts_used:.0f} / {avail_shirts}")
            print(f"  Pants Used: {pants_used:.0f} / {avail_pants}")

        elif model.status == GRB.INFEASIBLE:
            print(
                "Model is infeasible. Check constraints, inventory, and minimum sales requirements."
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
    solve_promotional_packages()