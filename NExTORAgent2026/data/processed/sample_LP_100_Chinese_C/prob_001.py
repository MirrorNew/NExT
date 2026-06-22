import gurobipy as gp
from gurobipy import GRB


def solve_feed_mix_problem():
    """
    Solves the animal feed mix optimization problem using Gurobi.
    The goal is to minimize the cost of feed while meeting daily nutritional requirements.
    A nonlinear-looking fixed cost is introduced:
    If total daily feed > 10 kg, an extra fixed cost F = 20 is charged; otherwise no extra cost.
    This is modeled via MILP with a binary variable.
    """
    try:
        # --- Data ---
        # Nutritional content per kg of feed: (Protein (g), Minerals (g), Vitamins (mg))
        # Price per kg of feed (¥/kg)

        feeds_data = {
            # Feed_ID: [Protein, Minerals, Vitamins, Price]
            1: [3, 1, 0.5, 0.2],
            2: [2, 0.5, 1, 0.7],
            3: [1, 0.2, 0.2, 0.4],
            4: [6, 2, 2, 0.3],
            5: [18, 0.5, 0.8, 0.8]
        }

        feed_ids = list(feeds_data.keys())

        # Daily nutritional requirements
        min_protein = 700  # g
        min_minerals = 30  # g
        min_vitamins = 100  # mg

        # Fixed cost parameters
        F = 20.0       # Fixed cost (¥) if total feed > 10 kg
        T = 10.0       # Threshold on total feed (kg)
        M = 1e4        # Big-M for linking binary and continuous vars (sufficiently large)

        # --- Create a new model ---
        model = gp.Model("FeedMixOptimization_With_FixedCost")

        # --- Decision Variables ---
        # x[i]: amount of feed i to use, in kilograms (kg)
        x = model.addVars(feed_ids, name="x", lb=0.0, vtype=GRB.CONTINUOUS)

        # Binary variable to indicate whether the fixed cost is incurred
        # y = 1 if total feed > 10 kg (allows exceeding 10 kg, fixed cost added)
        # y = 0 forces total feed <= 10 kg, no fixed cost
        y = model.addVar(name="y", vtype=GRB.BINARY)

        # Auxiliary variable for total feed
        total_feed = model.addVar(name="total_feed", lb=0.0, vtype=GRB.CONTINUOUS)

        # Link total_feed with x[i]
        model.addConstr(total_feed == gp.quicksum(x[i] for i in feed_ids),
                        name="TotalFeedDefinition")

        # --- Objective Function ---
        # Original linear objective
        # Cost = sum(price_i * x_i for i in feeds)
        # ❤ Non-linearity is introduced. ❤
        # total_cost = gp.quicksum(feeds_data[i][3] * x[i] for i in feed_ids)
        # New objective: variable feed cost + fixed cost F * y
        variable_cost = gp.quicksum(feeds_data[i][3] * x[i] for i in feed_ids)
        total_cost = variable_cost + F * y
        model.setObjective(total_cost, GRB.MINIMIZE)

        # --- Constraints ---

        # 1. Protein Requirement
        model.addConstr(
            gp.quicksum(feeds_data[i][0] * x[i] for i in feed_ids)
            >= min_protein, "ProteinRequirement")

        # 2. Minerals Requirement
        model.addConstr(
            gp.quicksum(feeds_data[i][1] * x[i] for i in feed_ids)
            >= min_minerals, "MineralsRequirement")

        # 3. Vitamins Requirement
        model.addConstr(
            gp.quicksum(feeds_data[i][2] * x[i] for i in feed_ids)
            >= min_vitamins, "VitaminsRequirement")

        # --- Nonlinear fixed-cost behavior modeled via linear constraints (Big-M) ---

        # We want: if total_feed <= T, then y = 0; if total_feed > T, then y = 1.
        # This is approximated with the following constraints:
        #
        # (1) total_feed <= T + M * y
        #     - If y = 0 ⇒ total_feed <= T   (cannot exceed threshold)
        #     - If y = 1 ⇒ total_feed <= T + M (essentially no upper binding from this)
        #
        # (2) total_feed >= T + ε * y
        #     - If y = 0 ⇒ total_feed >= 0   (no lower binding)
        #     - If y = 1 ⇒ total_feed >= T + ε (> T, so fixed cost applies only if we choose to exceed T)
        #
        # Here we take ε = 1e-4 to strictly enforce ">" in a numerical sense.

        eps = 1e-4

        # Upper link: when y=0, total_feed cannot exceed T
        model.addConstr(total_feed <= T + M * y, name="FixedCost_UpperLink")

        # Lower link: when y=1, total_feed must be at least T + eps
        model.addConstr(total_feed >= T + eps * y, name="FixedCost_LowerLink")

        # Suppress Gurobi output to console
        model.setParam('OutputFlag', 0)

        # Optimize the model
        model.optimize()

        # --- Results ---
        if model.status == GRB.OPTIMAL:
            print("Optimal feed mix found (with possible fixed cost).")
            print(f"Minimum Total Cost: {model.objVal:.2f} ¥")

            print("\nDecision on fixed cost:")
            if y.X > 0.5:
                print(f"  Fixed cost incurred (y = 1): +{F:.2f} ¥")
            else:
                print("  Fixed cost NOT incurred (y = 0)")

            print("\nAmount of each feed to use (kg):")
            total_feed_kg = 0.0
            for i in feed_ids:
                if x[i].X > 1e-6:  # Print only if the amount is significant
                    print(f"  Feed {i}: {x[i].X:.4f} kg")
                total_feed_kg += x[i].X
            print(f"\nTotal kilograms of feed: {total_feed_kg:.4f} kg")

            # Verification of nutritional intake
            achieved_protein = sum(feeds_data[i][0] * x[i].X for i in feed_ids)
            achieved_minerals = sum(feeds_data[i][1] * x[i].X
                                    for i in feed_ids)
            achieved_vitamins = sum(feeds_data[i][2] * x[i].X
                                    for i in feed_ids)
            print("\nNutritional Intake with this mix:")
            print(
                f"  Protein: {achieved_protein:.2f} g (Required: >= {min_protein} g)"
            )
            print(
                f"  Minerals: {achieved_minerals:.2f} g (Required: >= {min_minerals} g)"
            )
            print(
                f"  Vitamins: {achieved_vitamins:.2f} mg (Required: >= {min_vitamins} mg)"
            )

        elif model.status == GRB.INFEASIBLE:
            print(
                "Model is infeasible. The nutritional and fixed-cost-related requirements "
                "cannot be met with the given feeds under the specified constraints."
            )
        else:
            print(f"Optimization was stopped with status: {model.status}")

    except gp.GurobiError as e:
        print(f"Gurobi error code {e.errno}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    solve_feed_mix_problem()