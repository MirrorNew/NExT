def optimize_butcher_shops():
    from gurobipy import Model, GRB, quicksum
    import math

    # Create a new model
    m = Model("HotDogShopOptimization_Nonlinear")

    # Upper bound on number of large shops to consider (can be tuned)
    # With 30 workers and 4 workers per large shop, at most 7 large shops are possible (if no small shops).
    # We add a small buffer and set it to 10.
    L_max = 10

    # Decision variables: number of small shops (integer, non-negative)
    x = m.addVar(vtype=GRB.INTEGER, name="small_shops", lb=0)

    # ❤ Non-linearity is introduced. ❤
    # Original single integer variable for large shops is commented out and replaced with binary expansion.
    # y = m.addVar(vtype=GRB.INTEGER, name="large_shops", lb=0)

    # Binary variables z_i indicating whether the i-th large shop is built (1) or not (0)
    z = m.addVars(L_max, vtype=GRB.BINARY, name="large_shop_built")

    # The actual number of large shops y is the sum of z_i
    y = m.addVar(vtype=GRB.INTEGER, name="large_shops", lb=0, ub=L_max)

    # Link y with z_i
    m.addConstr(y == quicksum(z[i] for i in range(L_max)), name="link_y_z")

    # Set the objective: minimize total number of shops (small + large)
    m.setObjective(x + y, GRB.MINIMIZE)

    # ❤ Non-linearity is introduced. ❤
    # Original linear production constraint (constant 70 per large shop) is commented out
    # and replaced by an exponentially increasing production for each large shop.
    # m.addConstr(30 * x + 70 * y >= 500, name="production_constraint")

    # Precompute daily production for each potential i-th large shop:
    # shop i (1-based) produces 70 * 1.3^(i-1)
    prod_large = {i: 70.0 * (1.03 ** i) for i in range(L_max)}  # i=0..L_max-1 corresponds to shop index (i+1)

    # New (nonlinear-in-description, but modeled via fixed coefficients) production constraint
    # Total production from small shops + sum over chosen large shops' exponential production ≥ 500
    m.addConstr(
        30 * x + quicksum(prod_large[i] * z[i] for i in range(L_max)) >= 500,
        name="production_constraint_nonlinear"
    )

    # Add labor constraint: no more than 30 workers
    # Each small shop uses 2 workers; each large shop uses 4 workers
    m.addConstr(2 * x + 4 * y <= 30, name="labor_constraint")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Extract values
        num_small = int(x.X)
        num_large = int(y.X)
        total_shops = num_small + num_large

        # Optional: compute total production using the exponential formula
        total_prod = 30 * num_small + sum(
            prod_large[i] * z[i].X for i in range(L_max)
        )

        return {
            "total_shops": total_shops,
            "small_shops": num_small,
            "large_shops": num_large,
            "total_production": total_prod
        }
    else:
        # No feasible solution found
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_butcher_shops()
    if result is not None:
        print(f"Minimum Total Number of Shops: {result['total_shops']}")
        print(f"  Small shops: {result['small_shops']}")
        print(f"  Large shops: {result['large_shops']}")
        print(f"  Total daily production (approx.): {result['total_production']:.2f} hot dogs")
    else:
        print("No feasible solution found.")