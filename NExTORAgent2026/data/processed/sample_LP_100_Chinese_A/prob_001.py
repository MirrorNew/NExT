import gurobipy as gp
from gurobipy import GRB
import math


def solve_feed_mix_problem():
    """
    Solves the animal feed mix optimization problem using Gurobi.
    The goal is to minimize the total cost of feed over a 30‑day period
    while meeting daily nutritional requirements, where effective
    protein supply has a mild sinusoidal fluctuation over time.
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

        # Time horizon (days)
        days = range(1, 31)

        # Daily nutritional requirements
        min_protein = 700  # g
        min_minerals = 30  # g
        min_vitamins = 100  # mg

        # --- Create a new model ---
        model = gp.Model("FeedMixOptimization_Nonlinear_Protein_Fluctuation")

        # --- Decision Variables ---
        # Original (single-day) variables:
        # x[i]: amount of feed i to use, in kilograms (kg)
        # ❤ Non-linearity is introduced. ❤
        # x = model.addVars(feed_ids, name="x", lb=0.0, vtype=GRB.CONTINUOUS)

        # New variables: x[i, t] = amount of feed i used on day t (kg)
        x = model.addVars(feed_ids, days, name="x", lb=0.0, vtype=GRB.CONTINUOUS)

        # --- Objective Function ---
        # Original single-day objective:
        # Minimize the total cost of the feed mixture for one day
        # ❤ Non-linearity is introduced. ❤
        # total_cost = gp.quicksum(feeds_data[i][3] * x[i] for i in feed_ids)

        # New objective: minimize total cost over 30 days
        total_cost = gp.quicksum(
            feeds_data[i][3] * x[i, t] for i in feed_ids for t in days
        )
        model.setObjective(total_cost, GRB.MINIMIZE)

        # --- Constraints ---
        # For each day t, impose nutritional requirements

        for t in days:
            # Periodic adjustment factor for effective protein:
            # 1 + 0.1 * cos(π * t / 15)
            adj_factor = 1.0 + 0.1 * math.cos(math.pi * t / 15.0)

            # 1. Protein Requirement with periodic fluctuation
            # Effective protein on day t:
            #   (sum_i Protein_i * x[i, t]) * (1 + 0.1 * cos(π * t / 15)) >= 700
            # The adjustment factor is a known constant for each t, so this
            # constraint remains linear in x[i, t].
            model.addConstr(
                gp.quicksum(feeds_data[i][0] * x[i, t] for i in feed_ids)
                * adj_factor
                >= min_protein,
                name=f"ProteinRequirement_day{t}"
            )

            # 2. Minerals Requirement (linear, unchanged in form)
            model.addConstr(
                gp.quicksum(feeds_data[i][1] * x[i, t] for i in feed_ids)
                >= min_minerals,
                name=f"MineralsRequirement_day{t}"
            )

            # 3. Vitamins Requirement (linear, unchanged in form)
            model.addConstr(
                gp.quicksum(feeds_data[i][2] * x[i, t] for i in feed_ids)
                >= min_vitamins,
                name=f"VitaminsRequirement_day{t}"
            )

        # Suppress Gurobi output to console
        model.setParam('OutputFlag', 0)

        # Optimize the model
        model.optimize()

        # --- Results ---
        if model.status == GRB.OPTIMAL:
            print("Optimal 30-day feeding plan found.")
            print(f"Minimum Total Cost over 30 days: {model.objVal:.2f} ¥")

            # Print daily feed amounts (omit very small values)
            for t in days:
                print(f"\nDay {t}:")
                day_total_kg = 0.0
                for i in feed_ids:
                    val = x[i, t].X
                    if val > 1e-6:
                        print(f"  Feed {i}: {val:.4f} kg")
                        day_total_kg += val
                print(f"  Total feed: {day_total_kg:.4f} kg")

            # Optional: verify nutritional intake on each day
            print("\nVerification of daily nutritional intake (selected days):")
            for t in [1, 5, 10, 15, 20, 25, 30]:
                adj_factor = 1.0 + 0.1 * math.cos(math.pi * t / 15.0)
                protein_raw = sum(feeds_data[i][0] * x[i, t].X for i in feed_ids)
                protein_eff = protein_raw * adj_factor
                minerals = sum(feeds_data[i][1] * x[i, t].X for i in feed_ids)
                vitamins = sum(feeds_data[i][2] * x[i, t].X for i in feed_ids)
                print(f"\nDay {t}:")
                print(
                    f"  Effective Protein: {protein_eff:.2f} g "
                    f"(Required: >= {min_protein} g)"
                )
                print(
                    f"  Minerals: {minerals:.2f} g "
                    f"(Required: >= {min_minerals} g)"
                )
                print(
                    f"  Vitamins: {vitamins:.2f} mg "
                    f"(Required: >= {min_vitamins} mg)"
                )

        elif model.status == GRB.INFEASIBLE:
            print(
                "Model is infeasible. The nutritional requirements cannot be met "
                "with the given feeds under the specified constraints."
            )
            # Optionally compute IIS for debugging
            # model.computeIIS()
            # model.write("feed_mix_iis.ilp")
        else:
            print(f"Optimization was stopped with status: {model.status}")

    except gp.GurobiError as e:
        print(f"Gurobi error code {e.errno}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    solve_feed_mix_problem()