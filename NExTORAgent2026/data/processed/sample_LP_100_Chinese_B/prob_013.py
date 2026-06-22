import gurobipy as gp
from gurobipy import GRB


def solve_furniture_order_problem():
    try:
        # --- Problem Data ---
        # Costs per chair
        cost_per_chair_A = 50
        cost_per_chair_B = 45
        cost_per_chair_C = 40

        # Chairs per order
        chairs_per_order_A = 15
        chairs_per_order_B = 10
        chairs_per_order_C = 10

        # Cost per order
        cost_per_order_A = cost_per_chair_A * chairs_per_order_A
        cost_per_order_B = cost_per_chair_B * chairs_per_order_B
        cost_per_order_C = cost_per_chair_C * chairs_per_order_C

        # Chair limits
        min_total_chairs = 100
        max_total_chairs = 500  # still kept as data, but no longer enforced

        # --- Model Creation ---
        model = gp.Model("FurnitureOrderOptimization")

        # --- Decision Variables ---
        # Number of orders from each manufacturer (must be integer)
        orders_A = model.addVar(vtype=GRB.INTEGER, name="Orders_A", lb=0)
        orders_B = model.addVar(vtype=GRB.INTEGER, name="Orders_B", lb=0)
        orders_C = model.addVar(vtype=GRB.INTEGER, name="Orders_C", lb=0)

        # Binary variables to indicate if any order is placed with a manufacturer
        use_A = model.addVar(vtype=GRB.BINARY, name="Use_A")
        use_B = model.addVar(vtype=GRB.BINARY, name="Use_B")

        # --- Objective Function ---
        # Minimize the total cost of ordering
        total_cost = (orders_A * cost_per_order_A +
                      orders_B * cost_per_order_B +
                      orders_C * cost_per_order_C)
        model.setObjective(total_cost, GRB.MINIMIZE)

        # --- Constraints ---

        # 1. Calculate total number of chairs ordered
        total_chairs = (orders_A * chairs_per_order_A +
                        orders_B * chairs_per_order_B +
                        orders_C * chairs_per_order_C)

        # 2. Total chairs constraints
        model.addConstr(total_chairs >= min_total_chairs, "MinTotalChairs")
        # ❤ Non-linearity is introduced. ❤
        # model.addConstr(total_chairs <= max_total_chairs, "MaxTotalChairs")

        # 3. Link binary 'use' variables to the number of orders
        model.addConstr((use_A == 1) >> (orders_A >= 1), "Link_use_A_if_one")
        model.addConstr((use_A == 0) >> (orders_A == 0), "Link_use_A_if_zero")

        model.addConstr((use_B == 1) >> (orders_B >= 1), "Link_use_B_if_one")
        model.addConstr((use_B == 0) >> (orders_B == 0), "Link_use_B_if_zero")

        # 4. Conditional ordering constraints:
        # If the store decides to order from A, it must also order at least 10 chairs from B
        model.addConstr((use_A == 1) >> (orders_B >= 1),
                        "Conditional_A_implies_B")

        # If the store decides to order from B, it must also order from C
        model.addConstr((use_B == 1) >> (orders_C >= 1),
                        "Conditional_B_implies_C")

        # 5. Ratio constraint:
        # From manufacturer A: 15 * orders_A chairs
        # From manufacturers B and C combined: 10 * (orders_B + orders_C) chairs
        # Requirement:  (chairs from A) / (chairs from B + C) ≤ 3 / 2
        # i.e. 15*orders_A / (10*(orders_B + orders_C)) ≤ 3/2
        # Multiply both sides by 10*(orders_B + orders_C) to avoid division:
        # 15*orders_A * 2 ≤ 3 * 10 * (orders_B + orders_C)
        # 30 * orders_A ≤ 30 * (orders_B + orders_C)
        # which simplifies linearly to:
        # orders_A ≤ orders_B + orders_C
        # ❤ Non-linearity is introduced. ❤
        model.addConstr(orders_A <= orders_B + orders_C, "Ratio_A_to_BC_3_2")

        # --- Optimize Model ---
        model.optimize()

        # --- Results ---
        print("-" * 40)
        if model.status == GRB.OPTIMAL:
            print("Optimal solution found!")
            print(f"Minimum Total Cost: ${model.objVal:,.2f}")
            print("-" * 40)
            print("Order Plan:")
            print(f"  Orders from Manufacturer A: {orders_A.X:.0f}")
            print(f"    Chairs from A: {orders_A.X * chairs_per_order_A:.0f}")
            print(f"  Orders from Manufacturer B: {orders_B.X:.0f}")
            print(f"    Chairs from B: {orders_B.X * chairs_per_order_B:.0f}")
            print(f"  Orders from Manufacturer C: {orders_C.X:.0f}")
            print(f"    Chairs from C: {orders_C.X * chairs_per_order_C:.0f}")
            print("-" * 40)

            actual_total_chairs = (orders_A.X * chairs_per_order_A +
                                   orders_B.X * chairs_per_order_B +
                                   orders_C.X * chairs_per_order_C)
            print(f"Total Chairs Ordered: {actual_total_chairs:.0f}")
            print("-" * 40)

            print("Status of 'Use' Variables (for verification):")
            print(f"  Use Manufacturer A (use_A): {use_A.X:.0f}")
            print(f"  Use Manufacturer B (use_B): {use_B.X:.0f}")

        elif model.status == GRB.INFEASIBLE:
            print(
                "Model is infeasible. No solution exists that satisfies all constraints."
            )
            print(
                "Consider reviewing the constraints, especially the conditional ones and chair limits."
            )
        elif model.status == GRB.UNBOUNDED:
            print(
                "Model is unbounded. This typically means the objective can be improved indefinitely,"
            )
            print(
                "which might indicate an issue with constraints or objective direction for minimization."
            )
        else:
            print(f"Optimization was stopped with status: {model.status}")

    except gp.GurobiError as e:
        print(f"Gurobi error code {e.errno}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    solve_furniture_order_problem()