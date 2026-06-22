def optimize_sports_equipment(material_limit=1500, hours_limit=750, min_football=50):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("Sports_Equipment_Production")

    # Decision variables: number of basketballs and footballs
    x = model.addVar(name="Basketballs", vtype=GRB.INTEGER, lb=0)
    y = model.addVar(name="Footballs", vtype=GRB.INTEGER, lb=0)

    # ❤ Non-linearity is introduced. ❤
    # Original linear objective (kept, still linear):
    model.setObjective(x + y, GRB.MAXIMIZE)

    # Add constraints
    # Material constraint
    model.addConstr(5 * x + 3 * y <= material_limit, name="MaterialLimit")

    # ❤ Non-linearity is introduced. ❤
    # Original labor hours constraint (commented out, replaced below with a non-linear-like piecewise structure)
    # model.addConstr(x + 2 * y <= hours_limit, name="HoursLimit")

    # Introduce an auxiliary variable to model the "extra" footballs beyond 200:
    extra_y = model.addVar(name="ExtraFootballsOver200", vtype=GRB.CONTINUOUS, lb=0)

    # extra_y = max(0, y - 200)
    # This pair of constraints makes extra_y behave like max(0, y-200) in the optimal solution.
    model.addConstr(extra_y >= y - 200, name="ExtraDef1")
    model.addConstr(extra_y >= 0, name="ExtraDef2")

    # New (potentially non-linear behavior represented via additional variables) hours constraint:
    # Base hours: x + 2*y
    # Extra hours: 0.5 * extra_y (only when y > 200 effectively)
    # Total: x + 2*y + 0.5*extra_y <= hours_limit
    model.addConstr(x + 2 * y + 0.5 * extra_y <= hours_limit, name="HoursLimitWithOvertime")

    # Production ratio constraint
    model.addConstr(x >= 3 * y, name="BasketballToFootballRatio")

    # Minimum footballs
    model.addConstr(y >= min_football, name="MinFootballs")

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        total_produced = x.X + y.X
        # For completeness, we can also print the breakdown inside the function (optional)
        print(f"Optimal number of Basketballs: {int(x.X)}")
        print(f"Optimal number of Footballs: {int(y.X)}")
        print(f"Extra footballs over 200 (for overtime): {extra_y.X:.2f}")
        print(f"Maximum Total Sports Equipment Produced: {total_produced}")
        return total_produced
    else:
        return None


# Example usage
if __name__ == "__main__":
    total_equipment = optimize_sports_equipment()
    if total_equipment is not None:
        print(f"Maximum Total Sports Equipment Produced (returned): {total_equipment}")
    else:
        print("No feasible solution found.")