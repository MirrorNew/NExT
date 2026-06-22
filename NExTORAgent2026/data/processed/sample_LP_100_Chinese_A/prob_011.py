import gurobipy as gp
from gurobipy import GRB


def solve_promotional_packages():
    """
    Solves the promotional package problem to maximize revenue,
    subject to inventory and minimum sales constraints, with a
    non-linear interaction revenue term 0.1 * x * y * z.
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

        # Package prices (£ per package)
        prices = {'A': 30, 'B': 50}

        # Minimum sales requirements (packages)
        min_sales = {'A': 20, 'B': 10}

        # --- Create Gurobi Model ---
        model = gp.Model("PromotionalPackages_Nonlinear")

        # --- Decision Variables ---
        # N[p]: Number of packages of type p to sell
        N = model.addVars(packages,
                          name="NumPackages",
                          vtype=GRB.INTEGER,
                          lb=0)

        # New decision variable: promotion intensity level z (non-negative integer)
        z = model.addVar(name="PromotionIntensity",
                         vtype=GRB.INTEGER,
                         lb=0,ub=10)

        model.update()

        # --- Objective Function: Maximize Total Revenue ---
        # ❤ Non-linearity is introduced. ❤
        # model.setObjective(gp.quicksum(prices[p] * N[p] for p in packages),
        #                    GRB.MAXIMIZE)

        # Let x = N['A'], y = N['B'] for clarity
        x = N['A']
        y = N['B']

        # Total revenue = 30 * x + 50 * y + 0.1 * x * y * z

        Y = model.addVar(vtype=GRB.CONTINUOUS, name="Y")
        model.addConstr(Y ==  y * z, "Y")
        model.addConstr(z <= x)
        model.addConstr(z <= y)
        nonlinear_revenue = 30 * x + 50 * y +0.001 * x * Y
        model.setObjective(nonlinear_revenue, GRB.MAXIMIZE)

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

        # Suppress Gurobi output to console if desired
        # model.setParam('OutputFlag', 0)

        # Optimize the model
        model.optimize()

        # --- Results ---
        if model.status == GRB.OPTIMAL:
            print("Optimal package sales plan found (with non-linear revenue).")
            print(f"Maximum Total Revenue (including interaction term): £{model.ObjVal:.2f}")

            print("\nNumber of Packages to Sell:")
            for p in packages:
                print(f"  Package {p}: {N[p].X:.0f} units")

            print(f"\nOptimal promotion intensity level z: {z.X:.0f}")

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
            # Compute and print IIS (Irreducible Inconsistent Subsystem)
            # model.computeIIS()
            # model.write("promo_package_iis.ilp")
            # print("IIS written to promo_package_iis.ilp.")
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