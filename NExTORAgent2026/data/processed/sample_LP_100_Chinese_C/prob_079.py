def optimize_butcher_shops():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("HotDogShopOptimization_NonlinearRule")

    # Decision variables: number of small and large shops
    # Both are integers and non-negative
    x = m.addVar(vtype=GRB.INTEGER, name="small_shops", lb=0)
    y = m.addVar(vtype=GRB.INTEGER, name="large_shops", lb=0)

    # ❤ Non-linearity is introduced. ❤
    # Original objective (commented out):
    # m.setObjective(x + y, GRB.MINIMIZE)

    # -------- Introduce non-linearity via an additional binary variable --------
    # z = 1 if total daily production > 600, else 0
    z = m.addVar(vtype=GRB.BINARY, name="extra_cold_storage")

    # Total production expression
    total_prod = 30 * x + 70 * y

    # Big-M value large enough to cover maximum feasible production under 30 workers
    # Under labor constraint 2x + 4y <= 30, maximum production is bounded,
    # but we can safely choose a generous M (e.g., 3000) for linearization.
    M = 3000

    # Link z with "total_prod > 600"
    # If z = 0  => total_prod <= 600
    # If z = 1  => total_prod can exceed 600 (but must be >= 601 to make sense)
    m.addConstr(total_prod <= 600 + M * z, name="prod_upper_with_z")
    m.addConstr(total_prod >= 601 * z, name="prod_lower_with_z")

    # New objective: minimize number of real shops + 1 virtual shop if total_prod > 600
    m.setObjective(x + y + z, GRB.MINIMIZE)

    # Add production constraint: at least 500 hot dogs
    m.addConstr(total_prod >= 500, name="production_constraint")

    # Add labor constraint: no more than 30 workers
    m.addConstr(2 * x + 4 * y <= 30, name="labor_constraint")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal numbers of shops and the effective total counting the virtual shop
        optimal_small = int(x.X)
        optimal_large = int(y.X)
        virtual_shop = int(z.X)
        effective_total_shops = optimal_small + optimal_large + virtual_shop
        return {
            "small_shops": optimal_small,
            "large_shops": optimal_large,
            "virtual_shop": virtual_shop,
            "effective_total_shops": effective_total_shops,
            "objective_value": m.objVal,
            "total_production": int(total_prod.getValue())
        }
    else:
        # No feasible solution found
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_butcher_shops()
    if result is not None:
        print(f"Optimal number of small shops: {result['small_shops']}")
        print(f"Optimal number of large shops: {result['large_shops']}")
        print(f"Virtual shop used (1 if total production > 600): {result['virtual_shop']}")
        print(f"Effective total number of shops (including virtual): {result['effective_total_shops']}")
        print(f"Total daily production: {result['total_production']} hot dogs")
        print(f"Objective value reported by model: {result['objective_value']}")
    else:
        print("No feasible solution found.")