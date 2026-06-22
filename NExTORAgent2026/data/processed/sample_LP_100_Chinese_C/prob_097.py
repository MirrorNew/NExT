def optimize_bakery_production():
    from gurobipy import Model, GRB, quicksum

    # Create a new model
    m = Model("BakeryProduction")

    # Decision variables: number of brownies and lemon squares
    B = m.addVar(name="Brownies", vtype=GRB.INTEGER, lb=0)
    L = m.addVar(name="LemonSquares", vtype=GRB.INTEGER, lb=0)

    # Binary variable to indicate whether total production exceeds 400
    z = m.addVar(name="Over400", vtype=GRB.BINARY)

    # Auxiliary variable for total production
    T = m.addVar(name="TotalProduction", vtype=GRB.INTEGER, lb=0)

    # Link total production
    m.addConstr(T == B + L, name="Total_Production_Def")

    # Big-M value for production
    M = 2500  # large enough upper bound on total production (from chocolate capacity)

    # Logic: z = 1 if and only if T > 400
    # T <= 400 + M * z
    m.addConstr(T <= 400 + M * z, name="Upper_Bound_Over400")
    # T >= 401 * z
    m.addConstr(T >= 401 * z, name="Lower_Bound_Over400")

    # ❤ Non-linearity is introduced. ❤
    # m.setObjective(4 * B + 6 * L, GRB.MINIMIZE)

    # New objective with piecewise fiber consumption:
    # If T <= 400 (z = 0): 4B + 6L
    # If T > 400 (z = 1): (4+1)B + (6+0.5)L = 5B + 6.5L
    # Combine linearly: (4B + 6L) + z*(1B + 0.5L)
    m.setObjective(4 * B + 6 * L + z * (1 * B + 0.5 * L), GRB.MINIMIZE)

    # Add resource constraints
    m.addConstr(5 * B <= 2500, name="ChocolateConstraint")
    m.addConstr(7 * L <= 3300, name="LemonConstraint")

    # Lemon squares must be more than brownies
    # Note: Gurobi does not support strict inequalities directly.
    # To model L > B, we can use L >= B + 1
    m.addConstr(L >= B + 1, name="L_greater_than_B")

    # At least 40% of total items are brownies
    # The original constraint: 3B >= 2L
    m.addConstr(3 * B >= 2 * L, name="BrowniePercentage")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the optimal values: brownies, lemon squares, and minimal fiber usage
        return {
            "Brownies": int(B.X),
            "LemonSquares": int(L.X),
            "TotalFiber": m.objVal,
            "TotalProduction": int(T.X),
            "Over400Flag": int(z.X),
        }
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_bakery_production()
    if result is not None:
        print(f"Optimal Brownies: {result['Brownies']}")
        print(f"Optimal Lemon Squares: {result['LemonSquares']}")
        print(f"Total Production: {result['TotalProduction']}")
        print(f"Over 400 Flag (1=yes,0=no): {result['Over400Flag']}")
        print(f"Minimum Total Fiber Usage: {result['TotalFiber']}")
    else:
        print("No feasible solution found.")