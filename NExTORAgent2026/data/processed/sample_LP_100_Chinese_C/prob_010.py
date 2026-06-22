import gurobipy as gp
from gurobipy import GRB


def solve_tsp_mtz():
    """
    Solves the Traveling Salesman Problem (TSP) for 4 cities
    using the Miller-Tucker-Zemlin (MTZ) formulation for subtour elimination.

    Modified version:
    - Objective is to minimize total travel distance
      PLUS a fixed extra fee 50 if the arc from city 2 to city 3 is used
      at least once anywhere in the tour (mapping: 2 -> index 1, 3 -> index 2).
    """
    try:
        # --- Data ---
        # Number of cities
        num_cities = 4
        cities = range(num_cities)  # Indices 0, 1, 2, 3

        # Distance matrix d[i][j] is distance from city i to city j.
        # Mapping: City 1 -> 0, City 2 -> 1, City 3 -> 2, City 4 -> 3
        # Original Table:
        # | City |  1  |  2  |  3  |  4  |
        # | ---- | --- | --- | --- | --- |
        # | 1    | 0   | 10  | 20  | 12  |
        # | 2    | 10  | 0   | 5   | 10  |
        # | 3    | 20  | 5   | 0   | 8   |
        # | 4    | 15  | 12  | 8   | 0   |

        # Distances (costs) between cities. d[i][j] is cost from i to j.
        distances = {
            (0, 1): 10,
            (0, 2): 20,
            (0, 3): 12,  # From City 1 (index 0)
            (1, 0): 10,
            (1, 2): 5,
            (1, 3): 10,  # From City 2 (index 1)
            (2, 0): 20,
            (2, 1): 5,
            (2, 3): 8,   # From City 3 (index 2)
            (3, 0): 15,
            (3, 1): 12,
            (3, 2): 8    # From City 4 (index 3)
        }
        # Note: d[i,i] is implicitly infinity or such arcs are excluded.

        # --- Create Gurobi Model ---
        model = gp.Model("TSP_MTZ_with_extra_fee")

        # --- Decision Variables ---
        # x[i,j]: 1 if salesman travels from city i to city j, 0 otherwise
        x = model.addVars(cities, cities, vtype=GRB.BINARY, name="x")

        # u[i]: auxiliary variable for MTZ subtour elimination (position of city i in tour)
        # u[0] will be fixed to 1. Positions range from 1 to N.
        u = model.addVars(
            cities,
            vtype=GRB.CONTINUOUS,
            name="u",
            lb=1.0,
            ub=num_cities
        )

        # ❤ Non-linearity is introduced. ❤
        # y: binary variable that becomes 1 if the arc from city 2 to city 3 is used at least once.
        # Mapping: city 2 -> index 1, city 3 -> index 2, so this is arc (1, 2).
        y = model.addVar(vtype=GRB.BINARY, name="y_2_to_3_used")

        # --- Preprocessing: Remove x[i,i] variables (travel from a city to itself) ---
        for i in cities:
            x[i, i].ub = 0

        # --- Objective Function ---
        # Original objective (commented out) is pure distance minimization.
        # ❤ Non-linearity is introduced. ❤
        # model.setObjective(
        #     gp.quicksum(
        #         distances.get((i, j), 0) * x[i, j]
        #         for i in cities for j in cities if i != j
        #     ),
        #     GRB.MINIMIZE
        # )

        # New objective: distance + 50 * y
        model.setObjective(
            gp.quicksum(
                distances.get((i, j), 0) * x[i, j]
                for i in cities for j in cities if i != j
            ) + 50 * y,
            GRB.MINIMIZE
        )

        # --- Constraints ---
        # 1. Depart from each city exactly once
        for i in cities:
            model.addConstr(
                gp.quicksum(x[i, j] for j in cities if j != i) == 1,
                name=f"DepartOnce_{i}"
            )

        # 2. Enter each city exactly once
        for j in cities:
            model.addConstr(
                gp.quicksum(x[i, j] for i in cities if i != j) == 1,
                name=f"EnterOnce_{j}"
            )

        # 3. Subtour Elimination (MTZ constraints)
        # Let city 0 be the "depot" for sequencing purposes.
        model.addConstr(
            u[0] == 1,
            name="FixDepotOrder"  # Fix position of depot city (index 0)
        )

        for i in cities:
            if i == 0:
                continue  # Skip depot for these u bounds if u[0] is fixed differently
            model.addConstr(
                u[i] >= 2,
                name=f"u_lower_bound_{i}"
            )  # Positions 2 to N
            model.addConstr(
                u[i] <= num_cities,
                name=f"u_upper_bound_{i}"
            )

        for i in cities:
            for j in cities:
                if i == j or j == 0:  # Constraint is typically for non-depot j
                    continue
                # If x[i,j] = 1, then u[j] >= u[i] + 1
                # u[i] - u[j] + num_cities * x[i,j] <= num_cities - 1
                model.addConstr(
                    u[i] - u[j] + num_cities * x[i, j] <= num_cities - 1,
                    name=f"MTZ_{i}_{j}"
                )

        # 4. Link y with the use of arc (2 -> 3), i.e., (1 -> 2) in 0-based index.
        #    We want y = 1 if x[1,2] = 1; y = 0 if x[1,2] = 0.
        #    Because x[1,2] is already binary, x[1,2] == y ensures that relation.
        # ❤ Non-linearity is introduced. ❤
        model.addConstr(
            y >= x[1, 2],
            name="y_ge_x_1_2"
        )
        model.addConstr(
            y <= x[1, 2],
            name="y_le_x_1_2"
        )

        # Suppress Gurobi output to console if desired
        # model.setParam('OutputFlag', 0)

        # Optimize the model
        model.optimize()

        # --- Results ---
        if model.status == GRB.OPTIMAL:
            print("Optimal TSP route found.")
            print(f"Minimum Total Cost (distance + possible fee 50): {model.ObjVal:.2f}")
            print(f"Extra fee (50) charged? {'Yes' if y.X > 0.5 else 'No'}")

            print("\nOptimal Route (Original City Numbers):")
            # Reconstruct the path
            current_city_idx = 0  # Start from city 0 (Original City 1)
            route = [current_city_idx + 1]  # Store original city numbers

            for _ in range(num_cities - 1):
                found_next = False
                for j in cities:
                    if j != current_city_idx and x[current_city_idx, j].X > 0.5:
                        route.append(j + 1)
                        current_city_idx = j
                        found_next = True
                        break
                if not found_next:
                    print("Error: Could not reconstruct full route.")
                    break

            if len(route) == num_cities:
                # Check if it returns to start
                if x[current_city_idx, 0].X > 0.5:
                    route.append(route[0])  # Add starting city to show cycle
                    print(" -> ".join(map(str, route)))
                else:
                    print("Error: Route does not return to start or is incomplete.")
                    print("Partial route:", " -> ".join(map(str, route)))
            else:
                print("Error: Route reconstruction failed or incomplete.")
                print("Partial route:", " -> ".join(map(str, route)))

            print("\nSequence variables u_i (Original City Numbers):")
            for i in cities:
                print(f"  City {i+1}: u = {u[i].X:.1f}")

            print("\nArcs used (Original City Numbers):")
            for i in cities:
                for j in cities:
                    if i != j and x[i, j].X > 0.5:
                        print(
                            f"  {i+1} -> {j+1} "
                            f"(Distance: {distances.get((i, j), 0)})"
                        )

        elif model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Check constraints and data.")
        else:
            print(f"Optimization stopped with status: {model.status}")

    except gp.GurobiError as e:
        print(f"Gurobi error code {e.errno}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    solve_tsp_mtz()