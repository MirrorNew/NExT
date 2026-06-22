import gurobipy as gp
from gurobipy import GRB


def main():
    # -----------------------------
    # Problem data (from Parameters List)
    # -----------------------------
    factory_assignment = {
        '1': [2, 7, 12],
        '2': [6, 10],
        '3': [3, 9, 11],
        '4': [4, 5],
        '5': [1, 8]
    }

    order_assignment = {
        '1': 5,
        '2': 1,
        '3': 3,
        '4': 4,
        '5': 4,
        '6': 2,
        '7': 1,
        '8': 5,
        '9': 3,
        '10': 2,
        '11': 3,
        '12': 1
    }

    joint_orders = [10, 11, 12]
    standard_orders = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    factory_order_counts = {
        '1': 3,
        '2': 2,
        '3': 3,
        '4': 2,
        '5': 2
    }

    raw_cost_per_factory = {
        '1': 17.0,
        '2': 16.3,
        '3': 13.4,
        '4': 7.0,
        '5': 10.0
    }

    total_raw_cost = 63.7

    rebate_per_factory = {
        '1': 0.2,
        '2': 0.0,
        '3': 0.1,
        '4': 0.0,
        '5': 0.0
    }

    total_rebate = 0.3
    final_total_cost = 63.4

    min_orders_per_factory = 1
    max_orders_per_factory = 4
    rebate_threshold_orders = 3
    rebate_rate = 0.1

    # -----------------------------
    # Sets and parameters derived from data
    # -----------------------------
    factories = [1, 2, 3, 4, 5]
    orders = list(range(1, 13))

    # Cost matrix c_ij (only feasible entries from the original table)
    # Values not in this dict are treated as infeasible.
    c = {}

    # Factory 1
    c[1, 1] = 9
    c[1, 2] = 2
    c[1, 3] = 7
    c[1, 4] = 8
    c[1, 6] = 6
    c[1, 7] = 5
    c[1, 8] = 4
    c[1, 9] = 3
    c[1, 10] = 11.2
    c[1, 11] = 9.2
    c[1, 12] = 10

    # Factory 2
    c[2, 1] = 6
    c[2, 2] = 4
    c[2, 3] = 3
    c[2, 6] = 5
    c[2, 7] = 7
    c[2, 8] = 8
    c[2, 9] = 9
    c[2, 10] = 11.3
    c[2, 11] = 11.4
    c[2, 12] = 11.1

    # Factory 3
    c[3, 1] = 5
    c[3, 2] = 8
    c[3, 3] = 1
    c[3, 4] = 8
    c[3, 6] = 4
    c[3, 7] = 6
    c[3, 8] = 7
    c[3, 9] = 2
    c[3, 10] = 12
    c[3, 11] = 10.4
    c[3, 12] = 12.4

    # Factory 4
    c[4, 1] = 7
    c[4, 2] = 6
    c[4, 3] = 9
    c[4, 4] = 4
    c[4, 5] = 3
    c[4, 6] = 2
    c[4, 7] = 5
    c[4, 8] = 8
    c[4, 9] = 7
    c[4, 10] = 9.9
    c[4, 11] = 10.8
    c[4, 12] = 12.2

    # Factory 5
    c[5, 1] = 8
    c[5, 2] = 5
    c[5, 3] = 6
    c[5, 4] = 4
    c[5, 5] = 9
    c[5, 6] = 7
    c[5, 7] = 3
    c[5, 8] = 2
    c[5, 9] = 1
    c[5, 10] = 11.4
    c[5, 11] = 10.6
    c[5, 12] = 13.1

    feasible_pairs = list(c.keys())

    # Big-M value (large enough compared to costs)
    M = 100.0

    # -----------------------------
    # Build model
    # -----------------------------
    model = gp.Model("Haitong_Order_Factory_Assignment")

    # Decision variables
    # x_ij: 1 if factory i undertakes order j
    x = model.addVars(
        feasible_pairs,
        vtype=GRB.BINARY,
        name="x"
    )

    # n_i: number of orders undertaken by factory i
    n = model.addVars(
        factories,
        vtype=GRB.INTEGER,
        lb=min_orders_per_factory,
        ub=max_orders_per_factory,
        name="n"
    )

    # z_i: 1 if factory i has at least rebate_threshold_orders
    z = model.addVars(
        factories,
        vtype=GRB.BINARY,
        name="z"
    )

    # b_i: bonus for factory i
    b = model.addVars(
        factories,
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        name="b"
    )

    # m_ij: auxiliary variables for min-cost linearization
    m = model.addVars(
        feasible_pairs,
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        name="m"
    )

    # -----------------------------
    # Constraints
    # -----------------------------

    # Standard orders: sum_i x_ij = 1
    for j in standard_orders:
        model.addConstr(
            gp.quicksum(x[i, j] for i in factories if (i, j) in feasible_pairs) == 1,
            name=f"standard_assign_{j}"
        )

    # Joint orders: sum_i x_ij = 2
    for j in joint_orders:
        model.addConstr(
            gp.quicksum(x[i, j] for i in factories if (i, j) in feasible_pairs) == 2,
            name=f"joint_assign_{j}"
        )

    # Definition of n_i and load bounds (min/max) – n_i already has bounds
    for i in factories:
        model.addConstr(
            n[i] == gp.quicksum(x[i, j] for j in orders if (i, j) in feasible_pairs),
            name=f"def_n_{i}"
        )

    # Linking z_i and n_i (at least rebate_threshold_orders for bonus)
    for i in factories:
        # n_i >= threshold * z_i
        model.addConstr(
            n[i] >= rebate_threshold_orders * z[i],
            name=f"link_z_lower_{i}"
        )
        # n_i <= (threshold - 1) + (max_orders - (threshold - 1)) * z_i
        # Here: threshold = 3, max_orders = 4 => n_i <= 2 + 2 z_i
        model.addConstr(
            n[i] <= (rebate_threshold_orders - 1)
            + (max_orders_per_factory - (rebate_threshold_orders - 1)) * z[i],
            name=f"link_z_upper_{i}"
        )

    # m_ij >= c_ij - M (1 - x_ij)
    for (i, j) in feasible_pairs:
        model.addConstr(
            m[i, j] >= c[i, j] - M * (1 - x[i, j]),
            name=f"link_m_{i}_{j}"
        )

    # Bonus bounds via m_ij and indicator constraints:
    # For every feasible (i,j), when x_ij = 1:
    #   b_i <= rebate_rate * m_ij
    #   b_i >= rebate_rate * m_ij
    # This forces b_i = rebate_rate * m_ij for the (i,j) that attains the min cost.
    for (i, j) in feasible_pairs:
        # Upper bound when x_ij = 1
        model.addGenConstrIndicator(
            x[i, j], 1, b[i] <= rebate_rate * m[i, j],
            name=f"ind_b_ub_{i}_{j}"
        )
        # Lower bound when x_ij = 1
        model.addGenConstrIndicator(
            x[i, j], 1, b[i] >= rebate_rate * m[i, j],
            name=f"ind_b_lb_{i}_{j}"
        )

    # Bonus only if z_i = 1: b_i <= M * z_i
    for i in factories:
        model.addConstr(
            b[i] <= M * z[i],
            name=f"b_active_{i}"
        )

    # -----------------------------
    # Objective function
    # Minimize total cost - total bonus
    # -----------------------------
    assign_cost = gp.quicksum(
        c[i, j] * x[i, j] for (i, j) in feasible_pairs
    )
    total_bonus = gp.quicksum(b[i] for i in factories)

    model.setObjective(assign_cost - total_bonus, GRB.MINIMIZE)

    # -----------------------------
    # Solve
    # -----------------------------
    model.optimize()

    # -----------------------------
    # Output solution
    # -----------------------------
    if model.status == GRB.OPTIMAL:
        print("Optimal objective value (total cost with bonus):", model.objVal)
        print("Expected final_total_cost from Parameters List:", final_total_cost)
        print()

        print("Assignments (factory -> orders):")
        for i in factories:
            assigned_orders = [
                j for j in orders if (i, j) in feasible_pairs and x[i, j].X > 0.5
            ]
            print(f"  Factory {i}: orders {assigned_orders}")

        print("\nOrder-wise assignment:")
        for j in orders:
            assigned_factories = [
                i for i in factories if (i, j) in feasible_pairs and x[i, j].X > 0.5
            ]
            print(f"  Order {j}: factories {assigned_factories}")

        print("\nFactory statistics:")
        for i in factories:
            print(
                f"  Factory {i}: n_i = {n[i].X}, z_i = {int(round(z[i].X))}, "
                f"bonus b_i = {b[i].X:.4f}"
            )

        computed_raw_cost = sum(
            c[i, j] * x[i, j].X for (i, j) in feasible_pairs
        )
        computed_total_bonus = sum(b[i].X for i in factories)

        print("\nComputed raw cost:", round(computed_raw_cost, 4))
        print("Expected total_raw_cost from Parameters List:", total_raw_cost)
        print("Computed total bonus:", round(computed_total_bonus, 4))
        print("Expected total_rebate from Parameters List:", total_rebate)
    else:
        print("No optimal solution found. Model status:", model.status)


if __name__ == "__main__":
    main()