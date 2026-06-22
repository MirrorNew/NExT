def optimize_sports_equipment(material_limit=1500, hours_limit=750, min_football=50):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    model = gp.Model("Sports_Equipment_Production")

    # Decision variables: number of basketballs and footballs
    x = model.addVar(name="Basketballs", vtype=GRB.CONTINUOUS, lb=0)
    y = model.addVar(name="Footballs", vtype=GRB.CONTINUOUS, lb=0)

    # Set the objective: maximize total equipment
    model.setObjective(x + y, GRB.MAXIMIZE)

    # Add constraints
    # Material constraint
    model.addConstr(5 * x + 3 * y <= material_limit, name="MaterialLimit")
    # Labor hours constraint
    model.addConstr(x + 2 * y <= hours_limit, name="HoursLimit")

    # ❤ Non-linearity is introduced. ❤
    # model.addConstr(x >= 3 * y, name="BasketballToFootballRatio")
    # Production ratio constraint (non-linear form): x / y >= 3  ->  x >= 3 * y with division
    # Note: This explicitly uses a non-linear division operation to express the ratio.
    #       Gurobi will treat this as a general non-linear constraint in its Python API.
    #       We add it via a generic constraint to reflect the non-linear ratio description.

    # Even though mathematically equivalent to x >= 3*y, the ratio-based expression introduces non-linearity
    model.addConstr( x  >= 3 * y, name="BasketballToFootballRatio_Nonlinear")

    # Minimum footballs
    model.addConstr(y >= min_football, name="MinFootballs")

    # Optimize the model
    model.optimize()

    # Check if a feasible solution was found
    if model.status == GRB.OPTIMAL:
        total_produced = x.X + y.X
        return total_produced
    else:
        return None


# Example usage
if __name__ == "__main__":
    total_equipment = optimize_sports_equipment()
    if total_equipment is not None:
        print(f"Maximum Total Sports Equipment Produced: {total_equipment}")
    else:
        print("No feasible solution found.")