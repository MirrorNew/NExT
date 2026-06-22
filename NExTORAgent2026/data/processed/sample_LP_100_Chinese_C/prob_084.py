def optimize_snacks():
    from gurobipy import Model, GRB

    # Create a new model
    m = Model("Snack_Suitcases_Maximize")

    # Decision variables
    # x: number of small suitcases
    # y: number of large suitcases
    x = m.addVar(vtype=GRB.INTEGER, name="small_suitcases")
    y = m.addVar(vtype=GRB.INTEGER, name="large_suitcases")

    # ❤ Non-linearity is introduced. ❤
    # m.setObjective(50 * x + 80 * y, GRB.MAXIMIZE)
    # Introduce a binary variable to capture the fixed cost when y >= 31
    z = m.addVar(vtype=GRB.BINARY, name="large_over_30_indicator")

    # Big-M value for linking y and z (y <= 50, so 50 is sufficient)
    M = 50

    # If y >= 31, then z must be 1; if y <= 30, then z can be 0
    # y >= 31 * z
    m.addConstr(y >= 31 * z, name="large_over_30_lower")
    # y <= 30 + M * z  (when z=0 => y<=30; when z=1 => y<=30+M, which is non-binding)
    m.addConstr(y <= 30 + M * z, name="large_over_30_upper")

    # Objective: maximize total snacks minus penalty-equivalent of fixed cost
    # Note: Each snack is treated as 1 unit of "benefit"; a fixed cost of 2000
    # is modeled as subtracting 2000 units from the objective if z=1.
    m.setObjective(50 * x + 80 * y - 2000 * z, GRB.MAXIMIZE)

    # Add constraints
    m.addConstr(x - 2 * y >= 0, "preference_constraint")
    m.addConstr(x <= 70, "max_small")
    m.addConstr(y <= 50, "max_large")
    m.addConstr(y >= 15, "min_large")
    m.addConstr(x + y <= 70, "total_limit")

    # Optimize the model
    m.optimize()

    # Check if a feasible solution was found
    if m.status == GRB.OPTIMAL:
        # Return the maximum objective value and decision variables
        return {
            "max_effective_snacks_obj": m.objVal,
            "small_suitcases": int(x.X),
            "large_suitcases": int(y.X),
            "over_30_large": int(z.X),
        }
    else:
        # No feasible solution
        return None


# Example usage
if __name__ == "__main__":
    result = optimize_snacks()
    if result is not None:
        print(f"Maximum effective snacks objective value: {result['max_effective_snacks_obj']}")
        print(f"Number of small suitcases: {result['small_suitcases']}")
        print(f"Number of large suitcases: {result['large_suitcases']}")
        print(f"Large suitcases > 30 triggers fixed cost (z): {result['over_30_large']}")
    else:
        print("No feasible solution found.")